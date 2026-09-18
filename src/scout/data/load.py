"""Streaming the Inside Airbnb CSVs into PostgreSQL.

The shape of the load is dictated by three facts.

**The review file does not fit in memory.** London's is 2.2 million rows, so
both files are streamed with `csv.DictReader` and pushed straight into a staging
table with `COPY`; nothing accumulates in Python but the listing id set and the
lookup vocabulary.

**The most recent reviews are chosen in SQL.** Keeping the newest N per listing
in Python means tracking a heap per listing across the whole file; a window
function over the staged rows says the same thing in one statement and lets the
database do the sorting it is built for.

**A reload replaces, it does not duplicate.** Rows go into a temporary table
first and then upsert on the upstream listing id, so `listings.id` is stable
across loads -- reviews and, later, evaluation labels reference it. Listings
that have left the snapshot are deleted, so the database says what the file
says.

Personal fields never reach this module in a form that could be written: the
parsers hand back `ListingRow` and `ReviewRow`, which have no field for a host
or a reviewer, and the COPY writes those dataclasses column by column.
"""

from __future__ import annotations

import csv
import gzip
import logging
import time
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

from scout.config import Settings
from scout.data.parsers import (
    INT8,
    OTHER_PROPERTY_TYPE,
    REQUIRED_LISTING_COLUMNS,
    REQUIRED_REVIEW_COLUMNS,
    ListingRow,
    ReviewRow,
    RowParseError,
    canonical_spellings,
    collapse_property_type,
    kept_property_types,
    missing_columns,
    parse_int,
    parse_listing_row,
    parse_review_row,
)

logger = logging.getLogger(__name__)

Connection = psycopg.Connection[TupleRow]

# One review body is a paragraph, but the format allows a field far larger than
# csv's default 128 KB ceiling, and hitting it aborts the read. Raised once,
# with a bound, rather than removed.
_MAX_CSV_FIELD_BYTES = 16 * 1024 * 1024

_PROGRESS_EVERY = 250_000

_LISTING_COLUMNS = (
    "source_listing_id",
    "name",
    "description",
    "neighbourhood_id",
    "room_type_id",
    "property_type_id",
    "latitude",
    "longitude",
    "price_gbp",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "rating",
    "location_score",
    "cleanliness_score",
    "number_of_reviews",
    "instant_bookable",
    "host_is_superhost",
    "amenities",
)

_REVIEW_COLUMNS = ("source_listing_id", "date", "comments")

# Mirrors `listings` minus the columns the pipeline fills later: no generated
# id, no doc_text, no embedding. ON COMMIT DROP ties its lifetime to the load's
# transaction, so a failed load leaves nothing behind.
_CREATE_LISTINGS_STAGING = """
CREATE TEMP TABLE listings_staging (
    source_listing_id int8   NOT NULL,
    name              text   NOT NULL,
    description       text,
    neighbourhood_id  int4   NOT NULL,
    room_type_id      int2   NOT NULL,
    property_type_id  int2   NOT NULL,
    latitude          float8 NOT NULL,
    longitude         float8 NOT NULL,
    price_gbp         float4,
    accommodates      int2   NOT NULL,
    bedrooms          int2,
    beds              int2,
    bathrooms         float4,
    minimum_nights    int4   NOT NULL,
    rating            float4,
    location_score    float4,
    cleanliness_score float4,
    number_of_reviews int4   NOT NULL,
    instant_bookable  bool,
    host_is_superhost bool,
    amenities         text[] NOT NULL
) ON COMMIT DROP
"""

_CREATE_REVIEWS_STAGING = """
CREATE TEMP TABLE reviews_staging (
    source_listing_id int8 NOT NULL,
    date              date NOT NULL,
    comments          text NOT NULL
) ON COMMIT DROP
"""

# The upsert. Everything but the upstream id is overwritten, and doc_text and
# the embedding are dropped when an input to the embedded document changed --
# an embedding of text that no longer exists is worse than no embedding at all,
# because nothing downstream can tell that it is stale. This covers the listing
# side of the document only; review text feeds it too, and whether a changed
# review set should invalidate an embedding is the embed step's decision.
# DISTINCT ON, because a file that lists one listing twice would otherwise make
# ON CONFLICT DO UPDATE touch the same row twice in one statement, which
# PostgreSQL refuses -- one repeated id would fail the entire load. The first
# occurrence in file order wins, and the report says how many were dropped.
_UPSERT_LISTINGS = """
INSERT INTO listings (
    source_listing_id, name, description, neighbourhood_id, room_type_id,
    property_type_id, latitude, longitude, price_gbp, accommodates, bedrooms, beds,
    bathrooms, minimum_nights, rating, location_score, cleanliness_score,
    number_of_reviews, instant_bookable, host_is_superhost, amenities
)
SELECT DISTINCT ON (source_listing_id)
    source_listing_id, name, description, neighbourhood_id, room_type_id,
    property_type_id, latitude, longitude, price_gbp, accommodates, bedrooms, beds,
    bathrooms, minimum_nights, rating, location_score, cleanliness_score,
    number_of_reviews, instant_bookable, host_is_superhost, amenities
FROM listings_staging
ORDER BY source_listing_id, ctid
ON CONFLICT (source_listing_id) DO UPDATE SET
    name              = EXCLUDED.name,
    description       = EXCLUDED.description,
    neighbourhood_id  = EXCLUDED.neighbourhood_id,
    room_type_id      = EXCLUDED.room_type_id,
    property_type_id  = EXCLUDED.property_type_id,
    latitude          = EXCLUDED.latitude,
    longitude         = EXCLUDED.longitude,
    price_gbp         = EXCLUDED.price_gbp,
    accommodates      = EXCLUDED.accommodates,
    bedrooms          = EXCLUDED.bedrooms,
    beds              = EXCLUDED.beds,
    bathrooms         = EXCLUDED.bathrooms,
    minimum_nights    = EXCLUDED.minimum_nights,
    rating            = EXCLUDED.rating,
    location_score    = EXCLUDED.location_score,
    cleanliness_score = EXCLUDED.cleanliness_score,
    number_of_reviews = EXCLUDED.number_of_reviews,
    instant_bookable  = EXCLUDED.instant_bookable,
    host_is_superhost = EXCLUDED.host_is_superhost,
    amenities         = EXCLUDED.amenities,
    doc_text = CASE WHEN (
        listings.name, listings.description, listings.neighbourhood_id,
        listings.room_type_id, listings.property_type_id, listings.amenities
    ) IS DISTINCT FROM (
        EXCLUDED.name, EXCLUDED.description, EXCLUDED.neighbourhood_id,
        EXCLUDED.room_type_id, EXCLUDED.property_type_id, EXCLUDED.amenities
    ) THEN NULL ELSE listings.doc_text END,
    embedding = CASE WHEN (
        listings.name, listings.description, listings.neighbourhood_id,
        listings.room_type_id, listings.property_type_id, listings.amenities
    ) IS DISTINCT FROM (
        EXCLUDED.name, EXCLUDED.description, EXCLUDED.neighbourhood_id,
        EXCLUDED.room_type_id, EXCLUDED.property_type_id, EXCLUDED.amenities
    ) THEN NULL ELSE listings.embedding END
"""

# Only listings the file no longer carries at all. A listing that is still
# there but was rejected this run keeps its row: its generated id is referenced
# by reviews and, later, by relevance labels, and losing that over one blank
# field would be a worse answer than a stale attribute. The rejection is
# counted and reported either way.
_DELETE_DEPARTED_LISTINGS = """
DELETE FROM listings
WHERE NOT EXISTS (
    SELECT 1 FROM listings_staging s WHERE s.source_listing_id = listings.source_listing_id
)
AND NOT (listings.source_listing_id = ANY(%s::int8[]))
"""

_STAGED_LISTING_COUNTS = """
SELECT count(*), count(DISTINCT source_listing_id) FROM listings_staging
"""

# Counted in SQL rather than against a Python set of every id in the file: the
# listings table is already current by the time this runs, and the set was the
# one thing whose memory grew with the size of the snapshot.
_REVIEWS_WITHOUT_A_LISTING = """
SELECT count(*) FROM reviews_staging s
WHERE NOT EXISTS (
    SELECT 1 FROM listings l WHERE l.source_listing_id = s.source_listing_id
)
"""

# Reviews carry no upstream id, so there is no key to upsert on: the set is
# replaced outright. `ctid` breaks ties between reviews written on the same day
# by falling back to their order in the file, which makes the choice repeatable
# without storing the review id that would identify a reviewer.
_REPLACE_REVIEWS = """
INSERT INTO reviews (listing_id, date, comments)
SELECT l.id, staged.date, staged.comments
FROM (
    SELECT
        source_listing_id,
        date,
        comments,
        row_number() OVER (
            PARTITION BY source_listing_id ORDER BY date DESC, ctid DESC
        ) AS recency
    FROM reviews_staging
) AS staged
JOIN listings l ON l.source_listing_id = staged.source_listing_id
WHERE staged.recency <= %s
"""

_RECORD_SNAPSHOT = """
INSERT INTO data_snapshot (id, snapshot_date, city, listing_count, review_count)
VALUES (true, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    snapshot_date = EXCLUDED.snapshot_date,
    city          = EXCLUDED.city,
    listing_count = EXCLUDED.listing_count,
    review_count  = EXCLUDED.review_count,
    loaded_at     = now()
"""


class LoadError(RuntimeError):
    """The load could not run: a missing file, a changed schema, a dead database.

    Distinct from `RowParseError`, which is one unusable row among many and is
    counted and reported. This one ends the load, because every alternative
    ends with a database that silently holds less than it claims to.
    """


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What one load did, in the numbers worth checking afterwards."""

    snapshot_date: date
    city: str
    listings_seen: int
    listings_loaded: int
    listings_rejected: int
    listings_duplicated: int
    listings_removed: int
    listing_rejections: tuple[tuple[str, int], ...]
    reviews_seen: int
    reviews_rejected: int
    reviews_without_listing: int
    reviews_kept: int
    neighbourhoods: int
    room_types: int
    property_types: int
    property_types_collapsed: int
    amenities: int
    elapsed_s: float


@dataclass(slots=True)
class _RowStats:
    """How many rows a pass read and how many it had to refuse, by field."""

    seen: int = 0
    rejected: int = 0
    by_field: Counter[str] = field(default_factory=Counter)
    # Rejected rows whose upstream id was still readable. They are the listings
    # that are in the file but could not be stored, which is not the same thing
    # as a listing that has left the snapshot.
    rejected_ids: set[int] = field(default_factory=set)

    def record(self, exc: RowParseError, source_id: int | None = None) -> None:
        self.rejected += 1
        self.by_field[exc.field] += 1
        if source_id is not None:
            self.rejected_ids.add(source_id)

    def top_reasons(self, limit: int = 5) -> tuple[tuple[str, int], ...]:
        return tuple(self.by_field.most_common(limit))


@dataclass(frozen=True, slots=True)
class _Vocabulary:
    """The lookup values one pass over the listings file found.

    Collected before anything is written because two of them are decided
    globally rather than per row: which property types are common enough to
    keep, and which spelling of an amenity wins.
    """

    neighbourhoods: frozenset[str]
    room_types: frozenset[str]
    property_types: frozenset[str]
    kept_property_types: frozenset[str]
    collapsed_property_types: int
    amenity_canonical: Mapping[str, str]
    amenities: frozenset[str]


def load_data(settings: Settings) -> LoadReport:
    """Load the configured snapshot into the configured database.

    Raises:
        LoadError: if the snapshot date or a CSV is missing, if a file's columns
            are not the ones Scout reads, or if the database rejects the load.
    """
    snapshot_date = _require_snapshot_date(settings.snapshot_date)
    try:
        conn = psycopg.connect(settings.database_url)
    except psycopg.Error as exc:
        raise LoadError(f"could not connect to the database: {exc}") from exc
    with conn:
        return load_into(
            conn,
            listings_csv=settings.listings_csv,
            reviews_csv=settings.reviews_csv,
            snapshot_date=snapshot_date,
            city=settings.city,
            max_reviews_per_listing=settings.max_reviews_per_listing,
        )


def load_into(
    conn: Connection,
    *,
    listings_csv: Path,
    reviews_csv: Path,
    snapshot_date: date,
    city: str,
    max_reviews_per_listing: int,
) -> LoadReport:
    """Load both files into the schema `conn` sees, in one transaction.

    Either the whole snapshot lands or none of it does: a half-loaded database
    looks exactly like a small city.

    Raises:
        LoadError: if a file is missing, is missing a column Scout reads, holds
            no usable listing at all, or if the database rejects the load.
    """
    started = time.monotonic()
    _require_file(listings_csv, "listings")
    _require_file(reviews_csv, "reviews")

    vocabulary = _scan_listings(listings_csv)

    with conn.transaction():
        with _stage("creating the staging tables"):
            conn.execute(_CREATE_LISTINGS_STAGING)
            conn.execute(_CREATE_REVIEWS_STAGING)

        lookups = _upsert_lookups(conn, vocabulary)
        listing_stats = _copy_listings(conn, listings_csv, vocabulary, lookups)

        with _stage("counting the staged listings"):
            staged, distinct = conn.execute(_STAGED_LISTING_COUNTS).fetchone() or (0, 0)
        if not distinct:
            raise LoadError(
                f"{listing_stats.seen} listing rows read and none could be stored "
                f"({_describe(listing_stats.top_reasons())}); refusing to replace "
                f"the database with nothing"
            )
        duplicated = int(staged) - int(distinct)
        if duplicated:
            logger.warning("%d listing rows repeat an id already in the file", duplicated)

        with _stage("writing listings"):
            conn.execute(_UPSERT_LISTINGS)
            removed = conn.execute(
                _DELETE_DEPARTED_LISTINGS, (sorted(listing_stats.rejected_ids),)
            ).rowcount
        logger.info("listings: %d loaded, %d removed", distinct, removed)

        review_stats = _copy_reviews(conn, reviews_csv)
        with _stage("writing reviews"):
            without_listing = int((conn.execute(_REVIEWS_WITHOUT_A_LISTING).fetchone() or (0,))[0])
            # Nothing references reviews, and they carry no key to upsert on:
            # the snapshot's reviews replace whatever was there.
            conn.execute("TRUNCATE reviews")
            kept = conn.execute(_REPLACE_REVIEWS, (max_reviews_per_listing,)).rowcount
        if without_listing:
            logger.warning(
                "%d review rows name a listing that is not in this snapshot", without_listing
            )
        logger.info("reviews: %d kept, at most %d per listing", kept, max_reviews_per_listing)

        with _stage("recording the snapshot"):
            conn.execute(_RECORD_SNAPSHOT, (snapshot_date, city, distinct, kept))

    return LoadReport(
        snapshot_date=snapshot_date,
        city=city,
        listings_seen=listing_stats.seen,
        listings_loaded=int(distinct),
        listings_rejected=listing_stats.rejected,
        listings_duplicated=duplicated,
        listings_removed=removed,
        listing_rejections=listing_stats.top_reasons(),
        reviews_seen=review_stats.seen,
        reviews_rejected=review_stats.rejected,
        reviews_without_listing=without_listing,
        reviews_kept=kept,
        neighbourhoods=len(vocabulary.neighbourhoods),
        room_types=len(vocabulary.room_types),
        property_types=len(vocabulary.property_types),
        property_types_collapsed=vocabulary.collapsed_property_types,
        amenities=len(vocabulary.amenities),
        elapsed_s=time.monotonic() - started,
    )


def _scan_listings(path: Path) -> _Vocabulary:
    """Read the listings file once for the values the lookup tables need.

    A first pass, because two decisions need the whole file before any row can
    be written: a property type is only rare relative to every other listing,
    and the winning spelling of an amenity is the most common one.
    """
    neighbourhoods: set[str] = set()
    room_types: set[str] = set()
    property_type_counts: Counter[str] = Counter()
    amenity_counts: Counter[str] = Counter()

    stats = _RowStats()
    for listing in _parsed_listings(path, stats):
        neighbourhoods.add(listing.neighbourhood)
        room_types.add(listing.room_type)
        property_type_counts[listing.property_type] += 1
        amenity_counts.update(listing.amenities)

    kept = kept_property_types(property_type_counts)
    canonical = canonical_spellings(amenity_counts)
    logger.info(
        "scanned %d listings: %d neighbourhoods, %d room types, %d property types "
        "(%d collapsed into %s), %d amenities",
        stats.seen,
        len(neighbourhoods),
        len(room_types),
        len(property_type_counts),
        len(property_type_counts) - len(kept),
        OTHER_PROPERTY_TYPE,
        len(canonical),
    )
    return _Vocabulary(
        neighbourhoods=frozenset(neighbourhoods),
        room_types=frozenset(room_types),
        property_types=frozenset(
            collapse_property_type(name, kept) for name in property_type_counts
        ),
        kept_property_types=kept,
        collapsed_property_types=len(property_type_counts) - len(kept),
        amenity_canonical=canonical,
        amenities=frozenset(canonical.values()),
    )


def _upsert_lookups(conn: Connection, vocabulary: _Vocabulary) -> dict[str, dict[str, int]]:
    """Fill the lookup tables and read back name -> id for each.

    Existing rows are left alone rather than replaced: `amenities.is_indexed`
    records which amenities earned an indexed boolean column, and a reload must
    not quietly clear that.
    """
    ids: dict[str, dict[str, int]] = {}
    for table, names in (
        ("neighbourhoods", vocabulary.neighbourhoods),
        ("room_types", vocabulary.room_types),
        ("property_types", vocabulary.property_types),
        ("amenities", vocabulary.amenities),
    ):
        insert = sql.SQL(
            "INSERT INTO {} (name) SELECT unnest(%s::text[]) ON CONFLICT (name) DO NOTHING"
        ).format(sql.Identifier(table))
        select = sql.SQL("SELECT name, id FROM {}").format(sql.Identifier(table))
        with _stage(f"filling {table}"):
            conn.execute(insert, (sorted(names),))
            ids[table] = {
                str(name): int(row_id) for name, row_id in conn.execute(select).fetchall()
            }
    return ids


def _copy_listings(
    conn: Connection,
    path: Path,
    vocabulary: _Vocabulary,
    lookups: Mapping[str, Mapping[str, int]],
) -> _RowStats:
    """Stream the listings file into the staging table."""
    neighbourhoods = lookups["neighbourhoods"]
    room_types = lookups["room_types"]
    property_types = lookups["property_types"]
    canonical = vocabulary.amenity_canonical

    stats = _RowStats()
    copy_statement = sql.SQL("COPY listings_staging ({}) FROM STDIN").format(
        sql.SQL(", ").join(sql.Identifier(column) for column in _LISTING_COLUMNS)
    )
    with _stage("copying listings"), conn.cursor() as cur, cur.copy(copy_statement) as copy:
        for listing in _parsed_listings(path, stats):
            property_type = collapse_property_type(
                listing.property_type, vocabulary.kept_property_types
            )
            copy.write_row(
                (
                    listing.source_listing_id,
                    listing.name,
                    listing.description,
                    neighbourhoods[listing.neighbourhood],
                    room_types[listing.room_type],
                    property_types[property_type],
                    listing.latitude,
                    listing.longitude,
                    listing.price_gbp,
                    listing.accommodates,
                    listing.bedrooms,
                    listing.beds,
                    listing.bathrooms,
                    listing.minimum_nights,
                    listing.rating,
                    listing.location_score,
                    listing.cleanliness_score,
                    listing.number_of_reviews,
                    listing.instant_bookable,
                    listing.host_is_superhost,
                    [canonical[name.casefold()] for name in listing.amenities],
                )
            )
    if stats.rejected:
        logger.warning(
            "rejected %d of %d listing rows: %s",
            stats.rejected,
            stats.seen,
            _describe(stats.top_reasons()),
        )
    return stats


def _copy_reviews(conn: Connection, path: Path) -> _RowStats:
    """Stream the reviews file into the staging table, one row at a time.

    Every readable row is staged; which of them belong to a listing this
    database holds is a question the join answers afterwards, in SQL.
    """
    stats = _RowStats()
    copy_statement = sql.SQL("COPY reviews_staging ({}) FROM STDIN").format(
        sql.SQL(", ").join(sql.Identifier(column) for column in _REVIEW_COLUMNS)
    )
    with _stage("copying reviews"), conn.cursor() as cur, cur.copy(copy_statement) as copy:
        for review in _parsed_reviews(path, stats):
            copy.write_row((review.source_listing_id, review.date, review.comments))
    if stats.rejected:
        logger.warning(
            "rejected %d of %d review rows: %s",
            stats.rejected,
            stats.seen,
            _describe(stats.top_reasons()),
        )
    return stats


def _parsed_listings(path: Path, stats: _RowStats) -> Iterator[ListingRow]:
    """Every listing row that can be stored, counting the ones that cannot."""
    for raw in _csv_rows(path, REQUIRED_LISTING_COLUMNS):
        stats.seen += 1
        try:
            listing = parse_listing_row(raw)
        except RowParseError as exc:
            # One unusable row is data, not a failure: it is counted, its reason
            # is kept, and the report says how many there were.
            stats.record(exc, _readable_source_id(raw))
            logger.debug("listing row %d rejected: %s", stats.seen, exc)
            continue
        yield listing


def _parsed_reviews(path: Path, stats: _RowStats) -> Iterator[ReviewRow]:
    """Every review row that can be stored, counting the ones that cannot."""
    for raw in _csv_rows(path, REQUIRED_REVIEW_COLUMNS):
        stats.seen += 1
        if stats.seen % _PROGRESS_EVERY == 0:
            logger.info("  %d review rows read", stats.seen)
        try:
            review = parse_review_row(raw)
        except RowParseError as exc:
            stats.record(exc)
            logger.debug("review row %d rejected: %s", stats.seen, exc)
            continue
        yield review


def _readable_source_id(raw: Mapping[str, str | None]) -> int | None:
    """The upstream id of a row that could not be parsed, if it had a usable one."""
    try:
        return parse_int(raw.get("id"), field="id", fits=INT8)
    except RowParseError:
        return None


def _csv_rows(path: Path, required: frozenset[str]) -> Iterator[Mapping[str, str | None]]:
    """Stream one CSV, gzipped or not, as dictionaries.

    Raises:
        LoadError: if the file cannot be opened or read.
    """
    csv.field_size_limit(_MAX_CSV_FIELD_BYTES)
    try:
        if path.suffix == ".gz":
            with gzip.open(path, mode="rt", encoding="utf-8", newline="") as gzipped:
                yield from _dict_rows(gzipped, path, required)
        else:
            with path.open(encoding="utf-8", newline="") as plain:
                yield from _dict_rows(plain, path, required)
    except (OSError, EOFError, zlib.error) as exc:
        # A truncated download raises EOFError or zlib.error rather than
        # OSError, and an interrupted load should still say which file it was
        # reading instead of surfacing as a bare traceback.
        raise LoadError(f"{path} could not be read: {exc}") from exc


def _dict_rows(
    handle: Iterable[str], path: Path, required: frozenset[str]
) -> Iterator[Mapping[str, str | None]]:
    """Check the header, then hand back one dictionary per row.

    Raises:
        LoadError: if the file is not valid CSV or UTF-8, or has lost a column
            Scout reads -- Inside Airbnb renames columns between snapshots, and
            a renamed column would otherwise surface as every row being
            rejected for a different reason.
    """
    reader = csv.DictReader(handle)
    try:
        absent = missing_columns(reader.fieldnames, required)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise LoadError(f"{path} could not be read: {exc}") from exc
    if absent:
        raise LoadError(
            f"{path} has no {', '.join(absent)} column; the snapshot's columns "
            f"are not the ones this loader reads"
        )
    try:
        yield from reader
    except (UnicodeDecodeError, csv.Error) as exc:
        raise LoadError(f"{path} could not be read at line {reader.line_num}: {exc}") from exc


@contextmanager
def _stage(what: str) -> Iterator[None]:
    """Name the step a database error happened in, rather than only the SQL."""
    try:
        yield
    except psycopg.Error as exc:
        raise LoadError(f"{what} failed: {exc}") from exc


def _require_file(path: Path, what: str) -> None:
    if not path.is_file():
        raise LoadError(
            f"no {what} file at {path}; download the snapshot from Inside Airbnb "
            f"and point the configuration at it -- nothing here fetches it for you"
        )


def _require_snapshot_date(raw: str | None) -> date:
    """The snapshot's release date, which a load refuses to run without.

    Listing ids are not stable between Inside Airbnb releases, so data whose
    snapshot is unknown cannot be compared with a measurement or a relevance
    label taken from another one.
    """
    if not raw:
        raise LoadError(
            "the snapshot date is not set; it is the release the CSVs were "
            "downloaded from, and without it no measurement taken against this "
            "data can be tied to the listings it was taken on"
        )
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise LoadError(f"the snapshot date {raw!r} is not an ISO date (YYYY-MM-DD)") from exc


def _describe(reasons: tuple[tuple[str, int], ...]) -> str:
    return ", ".join(f"{field_name}: {count}" for field_name, count in reasons) or "no reasons"
