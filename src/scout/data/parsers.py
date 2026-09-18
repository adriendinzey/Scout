"""Turning Inside Airbnb's CSV text into the values Scout stores.

Everything here is pure: a string in, a value or an exception out. No file
handles, no database, no settings -- which is what lets the messy cases (a price
of `""`, `"Shared half-bath"`, an amenity list with two spellings of the same
thing) be tested exhaustively and cheaply, because that is where the bugs are.

Two rules shape the module.

**Personal fields are dropped here, not later.** `ListingRow` and `ReviewRow`
name every field Scout keeps, and the parsers build them from the source row by
reading only those names. There is no path where a host or reviewer name sits in
a dict that reaches SQL, so nothing downstream has to remember to remove it.

**A row that cannot be stored is rejected, not repaired.** A missing price is a
legitimate NULL; a missing neighbourhood is a listing that can never be filtered
or cited, and guessing one would quietly corrupt every number measured later.
The first raises nothing, the second raises `RowParseError`, and the loader
counts rejections and reports them rather than swallowing them.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Container, Mapping
from dataclasses import dataclass, fields
from datetime import date

# docs/DATA.md section 4. Named here so a test can assert that no parser output
# carries one of them, and so the list has a home in the code that does the
# dropping rather than only in prose.
DROPPED_PERSONAL_COLUMNS = frozenset(
    {
        "host_id",
        "host_name",
        "host_about",
        "host_url",
        "host_thumbnail_url",
        "host_picture_url",
        "host_profile_id",
        "host_profile_url",
        "listing_url",
        "reviewer_id",
        "reviewer_name",
    }
)

# Without these a listing cannot be stored at all, so their absence is a broken
# file rather than a bad row -- the loader checks the header and stops, instead
# of rejecting all ninety thousand listings one at a time.
REQUIRED_LISTING_COLUMNS = frozenset(
    {
        "id",
        "name",
        "description",
        "neighbourhood_cleansed",
        "room_type",
        "property_type",
        "latitude",
        "longitude",
        "price",
        "accommodates",
        "bedrooms",
        "beds",
        "bathrooms",
        "bathrooms_text",
        "minimum_nights",
        "number_of_reviews",
        "review_scores_rating",
        "review_scores_location",
        "review_scores_cleanliness",
        "instant_bookable",
        "host_is_superhost",
        "amenities",
    }
)

REQUIRED_REVIEW_COLUMNS = frozenset({"listing_id", "date", "comments"})

# Where a property type is too rare to be worth its own value. Below it, a
# filter on the type can only ever return a handful of rows out of ninety
# thousand, nothing in an evaluation set will be labelled against it, and it
# still costs a line in the grounding block that Parse sends on every query --
# which is the block this project deliberately keeps byte-identical and cached.
# At 50, London keeps 25 of its 91 types and collapses 0.9% of listings. The
# words themselves are not lost: the original type stays in the listing's name
# and description, which is what gets embedded.
MIN_PROPERTY_TYPE_LISTINGS = 50

OTHER_PROPERTY_TYPE = "Other"

# `price` arrives as "$1,234.00" -- Inside Airbnb prints a dollar sign whatever
# the local currency is. The symbols go; the number stays.
_CURRENCY_CHARACTERS = str.maketrans("", "", "$\u00a3\u20ac,\u00a0 ")

# "1 bath", "1.5 baths", "2 shared baths", "1 private bath".
_NUMBERED_BATH = re.compile(r"^(?P<count>\d+(?:\.\d+)?)\s+(?:shared\s+|private\s+)?baths?$")
# "Half-bath", "Shared half-bath", "Private half-bath". Half of a bathroom is
# the source's own reading: where both columns are populated, the numeric
# `bathrooms` column says 0.5 for these.
_HALF_BATH = re.compile(r"^(?:shared\s+|private\s+)?half-bath$")

_WHITESPACE = re.compile(r"\s+")

_TRUE_FALSE = {"t": True, "true": True, "f": False, "false": False}

# The widths of the columns these values are stored in. A value that does not
# fit aborts the COPY that carries it, taking the whole load down over one row,
# so it is refused here as the single bad row it is. London holds nothing near
# these; a damaged download or another city may.
INT2 = (-32_768, 32_767)
INT4 = (-2_147_483_648, 2_147_483_647)
INT8 = (-9_223_372_036_854_775_808, 9_223_372_036_854_775_807)
# float4 stops a little above 3.4e38; past that PostgreSQL raises out of range.
FLOAT4_MAX = 3.4028234663852886e38


class RowParseError(ValueError):
    """A source row that cannot become a listing or a review.

    Carries the column that was wrong as well as the message, because the
    loader reports rejections grouped by column: "3 listings rejected" without
    a reason is indistinguishable from a parser that has quietly stopped
    working, and grouping on the message text would group on the offending
    value too.
    """

    def __init__(self, message: str, *, field: str) -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True, slots=True)
class ListingRow:
    """One listing, reduced to the fields Scout is allowed to store.

    Field names match `listings` columns, except that the three lookup
    dimensions hold their text here and become integer ids in the loader, which
    is the only place that knows the lookup tables.
    """

    source_listing_id: int
    name: str
    description: str | None
    neighbourhood: str
    room_type: str
    property_type: str
    latitude: float
    longitude: float
    price_gbp: float | None
    accommodates: int
    bedrooms: int | None
    beds: int | None
    bathrooms: float | None
    minimum_nights: int
    rating: float | None
    location_score: float | None
    cleanliness_score: float | None
    number_of_reviews: int
    instant_bookable: bool | None
    host_is_superhost: bool | None
    amenities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReviewRow:
    """One review, reduced to what citing it needs.

    The upstream review id is dropped along with the reviewer: it resolves back
    to a reviewer's profile on the public site, which is the same reason
    `host_url` goes.
    """

    source_listing_id: int
    date: date
    comments: str


def missing_columns(header: object, required: frozenset[str]) -> list[str]:
    """Which required columns a CSV header does not have, sorted.

    Args:
        header: the field names `csv.DictReader` found, or None for an empty file.
        required: the columns the parser reads.
    """
    if header is None:
        return sorted(required)
    if not isinstance(header, list | tuple):
        raise TypeError(f"header must be a sequence of column names, got {type(header).__name__}")
    return sorted(required - {str(name) for name in header})


def parse_price(raw: str | None, *, field: str = "price") -> float | None:
    """A price like `"$1,234.00"` as a number, in the snapshot's own currency.

    Empty is None, not zero: about a third of London's listings carry no price,
    and a NULL satisfies no comparison, so they are excluded by a price filter
    rather than being made to look free.

    Raises:
        RowParseError: if a non-empty value is not a number, or is negative.
    """
    text = (raw or "").strip()
    if not text:
        return None
    value = parse_float(text.translate(_CURRENCY_CHARACTERS), field=field, fits=FLOAT4_MAX)
    if value is None:
        raise RowParseError(f"{field} {raw!r} is not a number", field=field)
    if value < 0:
        raise RowParseError(f"{field} {raw!r} is negative", field=field)
    return value


def parse_bathrooms(raw: str | None) -> float | None:
    """`bathrooms_text` as a number of bathrooms.

    Shared and private are not distinguished: `listings` has one numeric
    bathrooms column, and inventing a second one here would be storing a field
    nothing can filter on. "Half-bath" is 0.5, which is the source's own
    reading of it.

    Returns None for an empty value, which the loader falls back on the numeric
    `bathrooms` column to fill.

    Raises:
        RowParseError: if a non-empty value matches neither shape.
    """
    text = _WHITESPACE.sub(" ", (raw or "").strip().lower())
    if not text:
        return None
    if _HALF_BATH.match(text):
        return 0.5
    match = _NUMBERED_BATH.match(text)
    if match is None:
        raise RowParseError(
            f"bathrooms_text {raw!r} is not a bathroom count", field="bathrooms_text"
        )
    return float(match["count"])


def parse_bool(raw: str | None, *, field: str) -> bool | None:
    """The source's `t`/`f` as a bool. Empty is None -- "the snapshot is silent".

    Raises:
        RowParseError: if a non-empty value is neither true nor false.
    """
    text = (raw or "").strip().lower()
    if not text:
        return None
    if text not in _TRUE_FALSE:
        raise RowParseError(f"{field} {raw!r} is neither t nor f", field=field)
    return _TRUE_FALSE[text]


def parse_int(raw: str | None, *, field: str, fits: tuple[int, int] | None = None) -> int | None:
    """An integer column. Empty is None.

    Args:
        raw: the source text.
        field: the column name, for the error.
        fits: the range the storing column can hold, when there is one.

    Raises:
        RowParseError: if a non-empty value is not a whole number, or is too
            large for the column that has to hold it.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = int(text)
    except ValueError as exc:
        raise RowParseError(f"{field} {raw!r} is not a whole number", field=field) from exc
    if fits is not None and not fits[0] <= value <= fits[1]:
        raise RowParseError(f"{field} {value} does not fit the column that stores it", field=field)
    return value


def parse_float(raw: str | None, *, field: str, fits: float | None = None) -> float | None:
    """A float column. Empty is None.

    Args:
        raw: the source text.
        field: the column name, for the error.
        fits: the largest magnitude the storing column can hold, when there is
            one -- `float()` accepts "1e400", and the column does not.

    Raises:
        RowParseError: if a non-empty value is not a finite number, or is too
            large for the column that has to hold it. NaN and infinity are
            refused outright: they compare false against every filter, so a
            listing carrying one would be silently unfindable.
    """
    text = (raw or "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise RowParseError(f"{field} {raw!r} is not a number", field=field) from exc
    if not math.isfinite(value):
        raise RowParseError(f"{field} {raw!r} is not a finite number", field=field)
    if fits is not None and abs(value) > fits:
        raise RowParseError(f"{field} {value} does not fit the column that stores it", field=field)
    return value


def parse_date(raw: str | None, *, field: str = "date") -> date:
    """An ISO `YYYY-MM-DD` date.

    Raises:
        RowParseError: if the value is empty or not an ISO date.
    """
    text = (raw or "").strip()
    if not text:
        raise RowParseError(f"{field} is empty", field=field)
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise RowParseError(f"{field} {raw!r} is not an ISO date", field=field) from exc


def normalize_amenity(raw: str) -> str:
    """One amenity, with its spelling settled but its wording left alone.

    Unicode is folded to a single form and runs of whitespace collapse, so a
    curly apostrophe and a straight one, or one space and two, stop making two
    different amenities out of one.
    Case is deliberately kept: these strings are shown to a guest and embedded
    into the listing's document, and `normalize_amenity` is not the place that
    decides which spelling wins -- the loader does, from real counts.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", raw)).strip()


def parse_amenities(raw: str | None) -> tuple[str, ...]:
    """The amenity list as normalized names, in source order, deduplicated.

    Duplicates are compared case-insensitively, because a listing that says both
    "Wifi" and "WiFi" has one wifi.

    Raises:
        RowParseError: if the value is not a JSON list of strings.
    """
    text = (raw or "").strip()
    if not text:
        return ()
    try:
        values = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RowParseError(f"amenities is not JSON: {text[:60]!r}", field="amenities") from exc
    if not isinstance(values, list):
        raise RowParseError(
            f"amenities is a {type(values).__name__}, expected a list", field="amenities"
        )

    names: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise RowParseError(f"amenity {value!r} is not a string", field="amenities")
        name = normalize_amenity(value)
        if not name or name.casefold() in seen:
            continue
        _refuse_nul(name, field="amenities")
        seen.add(name.casefold())
        names.append(name)
    return tuple(names)


def canonical_spellings(counts: Mapping[str, int]) -> dict[str, str]:
    """Pick one spelling per amenity from how often each was written.

    Inside Airbnb's amenity strings are free text in places, so the same
    amenity arrives as "Sonos sound system", "SONOS sound system" and "SOnos
    sound system". Splitting one amenity across three lookup rows would also
    split its frequency count three ways, and that count is what the indexed
    amenity selection is ranked on.

    Args:
        counts: normalized spelling -> how many listings wrote it that way.

    Returns:
        Case-folded name -> the spelling to store. The most common spelling
        wins; ties break alphabetically, so the result does not depend on the
        order rows happened to be read in.
    """
    best: dict[str, tuple[int, str]] = {}
    for spelling, count in counts.items():
        key = spelling.casefold()
        current = best.get(key)
        if current is None or (-count, spelling) < (-current[0], current[1]):
            best[key] = (count, spelling)
    return {key: spelling for key, (_, spelling) in best.items()}


def kept_property_types(
    counts: Mapping[str, int], *, min_listings: int = MIN_PROPERTY_TYPE_LISTINGS
) -> frozenset[str]:
    """The property types common enough to stay their own value.

    Args:
        counts: property type -> how many listings have it.
        min_listings: the threshold; see MIN_PROPERTY_TYPE_LISTINGS for why.
    """
    return frozenset(name for name, count in counts.items() if count >= min_listings)


def collapse_property_type(name: str, kept: Container[str]) -> str:
    """`name` if it survived the rare-type threshold, otherwise "Other"."""
    return name if name in kept else OTHER_PROPERTY_TYPE


def parse_listing_row(raw: Mapping[str, str | None]) -> ListingRow:
    """One source listing row as the fields Scout stores.

    Reads only permitted columns, so host identity never reaches the result --
    the caller's dict still holds it, this function's output cannot.

    Raises:
        RowParseError: if a column Scout stores NOT NULL is missing or
            unusable. Optional columns that are simply absent become None.
    """
    source_listing_id = _required_int(raw.get("id"), field="id", fits=INT8)
    name = _required_text(raw.get("name"), field="name")
    neighbourhood = _required_text(
        raw.get("neighbourhood_cleansed"), field="neighbourhood_cleansed"
    )
    room_type = _required_text(raw.get("room_type"), field="room_type")
    property_type = _required_text(raw.get("property_type"), field="property_type")

    latitude = _required_float(raw.get("latitude"), field="latitude")
    longitude = _required_float(raw.get("longitude"), field="longitude")
    if not -90.0 <= latitude <= 90.0:
        raise RowParseError(f"latitude {latitude} is out of range", field="latitude")
    if not -180.0 <= longitude <= 180.0:
        raise RowParseError(f"longitude {longitude} is out of range", field="longitude")

    accommodates = _required_int(raw.get("accommodates"), field="accommodates", fits=INT2)
    if accommodates < 1:
        raise RowParseError(
            f"accommodates {accommodates} is not a party of at least one", field="accommodates"
        )
    minimum_nights = _required_int(raw.get("minimum_nights"), field="minimum_nights", fits=INT4)
    if minimum_nights < 1:
        raise RowParseError(
            f"minimum_nights {minimum_nights} is less than a night", field="minimum_nights"
        )
    number_of_reviews = _required_int(
        raw.get("number_of_reviews"), field="number_of_reviews", fits=INT4
    )
    if number_of_reviews < 0:
        raise RowParseError(
            f"number_of_reviews {number_of_reviews} is negative", field="number_of_reviews"
        )

    # The text column is populated for all but a fraction of a percent of
    # listings and the numeric one for only two thirds, so the text leads and
    # the number fills in behind it.
    bathrooms = parse_bathrooms(raw.get("bathrooms_text"))
    if bathrooms is None:
        bathrooms = parse_float(raw.get("bathrooms"), field="bathrooms", fits=FLOAT4_MAX)

    return ListingRow(
        source_listing_id=source_listing_id,
        name=name,
        description=_optional_text(raw.get("description"), field="description"),
        neighbourhood=neighbourhood,
        room_type=room_type,
        property_type=property_type,
        latitude=latitude,
        longitude=longitude,
        price_gbp=parse_price(raw.get("price")),
        accommodates=accommodates,
        bedrooms=parse_int(raw.get("bedrooms"), field="bedrooms", fits=INT2),
        beds=parse_int(raw.get("beds"), field="beds", fits=INT2),
        bathrooms=bathrooms,
        minimum_nights=minimum_nights,
        rating=parse_float(
            raw.get("review_scores_rating"), field="review_scores_rating", fits=FLOAT4_MAX
        ),
        location_score=parse_float(
            raw.get("review_scores_location"), field="review_scores_location", fits=FLOAT4_MAX
        ),
        cleanliness_score=parse_float(
            raw.get("review_scores_cleanliness"), field="review_scores_cleanliness", fits=FLOAT4_MAX
        ),
        number_of_reviews=number_of_reviews,
        instant_bookable=parse_bool(raw.get("instant_bookable"), field="instant_bookable"),
        host_is_superhost=parse_bool(raw.get("host_is_superhost"), field="host_is_superhost"),
        amenities=parse_amenities(raw.get("amenities")),
    )


def parse_review_row(raw: Mapping[str, str | None]) -> ReviewRow:
    """One source review row as a citation: which listing, when, and what it says.

    The reviewer's name and id are never read, and neither is the review id.

    Raises:
        RowParseError: if the listing id, the date, or the text is unusable.
            An empty review body is a rejection: it can be neither cited nor
            embedded, and storing it would inflate the count of reviews a
            listing has to show for itself.
    """
    comments = _required_text(raw.get("comments"), field="comments")
    return ReviewRow(
        source_listing_id=_required_int(raw.get("listing_id"), field="listing_id"),
        date=parse_date(raw.get("date")),
        comments=comments,
    )


def listing_row_field_names() -> frozenset[str]:
    """Every field `ListingRow` carries. Used by the privacy test."""
    return frozenset(field.name for field in fields(ListingRow))


def review_row_field_names() -> frozenset[str]:
    """Every field `ReviewRow` carries. Used by the privacy test."""
    return frozenset(field.name for field in fields(ReviewRow))


def _optional_text(raw: str | None, *, field: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    _refuse_nul(text, field=field)
    return text


def _required_text(raw: str | None, *, field: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise RowParseError(f"{field} is empty", field=field)
    _refuse_nul(text, field=field)
    return text


def _refuse_nul(text: str, *, field: str) -> None:
    """No PostgreSQL text column can hold a NUL byte.

    Stripping it would edit a listing's own words on the way past, and the row
    would then differ from the file it was loaded from with nothing saying so.
    """
    if "\x00" in text:
        raise RowParseError(f"{field} contains a NUL byte, which no text column holds", field=field)


def _required_int(raw: str | None, *, field: str, fits: tuple[int, int] | None = None) -> int:
    value = parse_int(raw, field=field, fits=fits)
    if value is None:
        raise RowParseError(f"{field} is empty", field=field)
    return value


def _required_float(raw: str | None, *, field: str, fits: float | None = None) -> float:
    value = parse_float(raw, field=field, fits=fits)
    if value is None:
        raise RowParseError(f"{field} is empty", field=field)
    return value
