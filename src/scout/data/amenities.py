"""Amenity groups: which raw strings mean the same thing, and which earn a column.

Inside Airbnb's amenity list is free text in places. One London snapshot holds
8,735 distinct spellings for roughly a few hundred actual amenities, because the
source emits a *detail* alongside the thing itself: "Wifi" and "Fast wifi -- 73
Mbps" are one amenity measured twice, "Free dryer -- In unit" and "Dryer" are one
appliance described at two levels of confidence, and the long tail is hosts
typing their own ("Lamona Built In Single Fan Oven and Grill ... oven").

Two separate jobs live here, and keeping them apart is what makes the second one
honest.

**Grouping.** A group is one guest-facing amenity plus every spelling that means
it. Each spelling is matched against a group's patterns as a *whole string*, not
a substring: "Kitchen" is a kitchen, "There is a fridge in the kitchen for your
use refrigerator" is a refrigerator a host described in prose, and a substring
test cannot tell them apart. The merges are deliberately narrow where the source
distinguishes price or place -- free parking is never paid parking, an in-unit
washer and a shared one are one appliance but a laundromat down the road is not.

**Selection.** An index key column is a scarce resource: PostgreSQL allows 32 per
index, the embedding and the fixed filter columns spend 17, so 15 are left for
amenities. A group earns one when a guest asks for it *and* filtering on it after
the ranked scan would cost real recall. Those pull in opposite directions: wifi
is in 97% of listings -- asked for constantly, excludes almost nothing -- so
applying it to the rows a scan already returned is nearly free; pets allowed is
in 13%, which reduces a 100-candidate pool to about 13 rows, the difference
between an answer and "nothing matched". So the column goes to the
selective ones guests actually ask about, and the near-universal ones stay in
`listings.amenities` where they cost nothing to check.

Nothing in the grouping or counting here performs I/O, so every rule is testable
against real strings without a database. The two functions at the bottom that do
touch the database are the ones that write the decision into the schema.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from typing import Final

import psycopg
from psycopg import sql
from psycopg.rows import TupleRow

Connection = psycopg.Connection[TupleRow]

# How many spellings a group's patterns are tried against before the answer is
# reused. One snapshot holds ~9,000 distinct amenity strings and ~3.2 million
# occurrences of them, so without the cache every listing re-runs every pattern.
# Bounded rather than unbounded: a later snapshot adds spellings, and a cache
# that can only grow is a leak in a long-lived process.
_SPELLING_CACHE: Final = 32_768

_WHITESPACE: Final = re.compile(r"\s+")
# The source writes the same separator as an en dash, an em dash and a hyphen
# ("Private backyard -- Fully fenced" against "Backyard - Fully fenced"), and the
# same possessive with a curly or a straight quote. Folding them means one
# pattern per group instead of one per punctuation accident.
# Spelled as escapes, not as the characters themselves: this is the one place
# the punctuation is the subject rather than the text, and a reader cannot
# tell an en dash from a hyphen by looking at it.
_DASHES: Final = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-"})
_QUOTES: Final = str.maketrans({"\u2018": "'", "\u2019": "'"})
# Only a dash with space beside it separates an amenity from its detail. The
# hyphen inside "Self check-in" is part of the word, and normalizing it as a
# separator would split the amenity into three and match nothing.
_SEPARATOR: Final = re.compile(r"\s+-\s*|\s*-\s+")


def fold_amenity(raw: str) -> str:
    """The form a group's patterns are matched against.

    Case, Unicode form, whitespace runs and the three dash spellings are all
    accidents of how a listing was written rather than differences in what it
    offers. `normalize_amenity` already settles the first two for stored values;
    this repeats them so a pattern can be tested against a string straight out
    of the source file.
    """
    folded = unicodedata.normalize("NFKC", raw).casefold()
    folded = folded.translate(_DASHES).translate(_QUOTES)
    # " - " however it was written, so one pattern matches "Crib - available
    # upon request", "Free dryer <en dash> In unit" and "Backyard- Fully fenced".
    folded = _SEPARATOR.sub(" - ", folded)
    return _WHITESPACE.sub(" ", folded).strip()


@dataclass(frozen=True, slots=True)
class AmenityGroup:
    """One guest-facing amenity, its boolean column, and what counts as it.

    `patterns` are matched against the whole folded spelling (`fold_amenity`),
    which is what keeps a host's free text out of a group it merely mentions.
    `indexed` records the operator's decision: only an indexed group gets a
    column in the schema and a key slot in the index.
    """

    column: str
    name: str
    patterns: tuple[str, ...]
    indexed: bool
    # Why this group exists at all, and why the merge is safe -- the amenity set
    # is baked into the index, so the reasoning has to outlive the session that
    # chose it.
    why: str

    @property
    def regex(self) -> re.Pattern[str]:
        return _compiled(self.patterns)

    def matches(self, raw: str) -> bool:
        """True when `raw` is one of this group's spellings."""
        return self.regex.fullmatch(fold_amenity(raw)) is not None


@cache
def _compiled(patterns: tuple[str, ...]) -> re.Pattern[str]:
    """One alternation per group, compiled once.

    Cached on the pattern tuple rather than built at import so the group table
    stays plain data, and so a test can add a group without a second mechanism.
    """
    return re.compile("|".join(f"(?:{pattern})" for pattern in patterns))


# A number of spaces, bedrooms or Mbps the source appends to an amenity. It
# never changes whether the amenity is there, only how much of it there is.
_DETAIL: Final = r"(?: - \d+ (?:space|spaces|mbps))?"
# Airbnb asks hosts whether a baby item is there now or can be fetched. Both are
# a cot the guest can use, so both count; "paid" is kept out of nothing here
# because a cot that costs extra is still a cot (unlike parking, where free
# against paid is the question being asked).
_AVAILABILITY: Final = r"(?: - (?:available upon request|always at the listing))?"

# Every group the counts were ranked on, in the order the report prints when two
# share a prevalence. `indexed` is the confirmed selection; the rest are kept
# because the report's job is to show what was considered and rejected, not only
# what won -- and because a later snapshot can move a group across the line.
AMENITY_GROUPS: Final[tuple[AmenityGroup, ...]] = (
    # ------------------------------------------------- indexed: the 13 columns --
    AmenityGroup(
        column="has_dedicated_workspace",
        name="Dedicated workspace",
        patterns=("dedicated workspace",),
        indexed=True,
        why=(
            "The whole 'good for working remotely' class of query turns on this "
            "one flag, and at 44.5% of listings it halves the corpus."
        ),
    ),
    AmenityGroup(
        column="has_self_checkin",
        name="Self check-in",
        patterns=("self check-in",),
        indexed=True,
        why=(
            "Asked for by anyone arriving late (40.8%). The lock hardware the "
            "source lists beside it -- lockbox, keypad, smart lock -- is kept "
            "out, and costs nothing: exactly one listing in the snapshot has "
            "the hardware without the flag. 'Building staff' and 'Doorman' are "
            "a person handing over a key, which is the opposite of checking "
            "yourself in."
        ),
    ),
    AmenityGroup(
        column="has_bathtub",
        name="Bathtub",
        patterns=("bathtub",),
        indexed=True,
        why=(
            "A guest asking for a bath means the fixture, and the numeric "
            "`bathrooms` column counts rooms without saying what is in them. "
            "36.9% of listings."
        ),
    ),
    AmenityGroup(
        column="has_dryer",
        name="Dryer",
        patterns=(rf"(?:free |paid )?dryer(?: - in (?:unit|building))?{_DETAIL}",),
        indexed=True,
        why=(
            "Free against paid and in-unit against in-building are collapsed: "
            "the question is whether clothes can be dried. At 36.9% it also "
            "carries the whole laundry pair's selectivity, because a dryer "
            "nearly always comes with a washer -- 0.4% of listings have one "
            "without the other -- so 'washer and dryer' pushes down here and "
            "the washer costs almost nothing after the scan. 'Hair dryer' is a "
            "different appliance entirely, and a whole-string match is what "
            "keeps it out: a substring test would put 68% of listings here."
        ),
    ),
    AmenityGroup(
        column="has_dishwasher",
        name="Dishwasher",
        patterns=("dishwasher",),
        indexed=True,
        why=(
            "Selective at 36.2% and commonly asked for on longer stays. "
            "Whole-string matching is what keeps it out of the washer group."
        ),
    ),
    AmenityGroup(
        column="has_outdoor_space",
        name="Patio or balcony",
        patterns=(
            # "Patio or balcony" is the spelling the snapshot actually
            # publishes; the bare words and "terrace" match nothing in it and
            # are here for the snapshot that spells it differently. They cannot
            # widen the group wrongly, because a pattern has to match the whole
            # string.
            r"(?:private |shared )?(?:patio or balcony|patio|balcony|terrace)",
            r"(?:private |shared )?backyard(?: - (?:fully|not fully) fenced)?",
        ),
        indexed=True,
        why=(
            "One group for 'is there outside space', because the source splits "
            "it across patio, balcony and backyard. It never says 'garden' -- "
            "the guest's word for this -- except in 'Garden view' and "
            "'Courtyard view', which are a window rather than somewhere to sit "
            "and which whole-string matching drops. Private (15.2%) and shared "
            "or unqualified (18.3%) are merged into one 31.6% group: a shared "
            "yard is still outside space."
        ),
    ),
    AmenityGroup(
        column="has_free_parking",
        name="Free parking on premises",
        patterns=(
            r"free (?:driveway parking|parking garage|residential garage|carport|"
            rf"parking lot|parking) on premises{_DETAIL}",
        ),
        indexed=True,
        why=(
            "Every paid spelling is excluded: merging them would make the column "
            "a lie in the one direction a guest would notice. 'Free street "
            "parking' is excluded too -- it is a fact about the street, not a "
            "space the property can promise -- so it stays an array post-filter "
            "under its own name. The column holds 20.8% of listings; counting "
            "free street parking too would make it 25.8%."
        ),
    ),
    AmenityGroup(
        column="has_private_entrance",
        name="Private entrance",
        patterns=("private entrance",),
        indexed=True,
        why=("The usual proxy for privacy in a shared building, and selective at 24.8%."),
    ),
    AmenityGroup(
        column="has_elevator",
        name="Elevator",
        patterns=("elevator",),
        indexed=True,
        why=(
            "The source has no accessibility fields in this snapshot, so this is "
            "the only structural signal for a guest who cannot manage stairs. At "
            "18.4% it is both asked for and selective."
        ),
    ),
    AmenityGroup(
        column="has_crib",
        name="Crib",
        patterns=(
            rf"(?:paid )?crib{_AVAILABILITY}",
            rf"(?:paid )?pack 'n play/travel crib{_AVAILABILITY}",
        ),
        indexed=True,
        why=(
            "'Somewhere for the baby to sleep' is a hard requirement, not a "
            "preference, which is what makes 18.5% worth a column. A travel cot is "
            "the same answer to the guest's question, and one that has to be "
            "requested still counts -- the alternative is nothing to sleep in."
        ),
    ),
    AmenityGroup(
        column="has_pets_allowed",
        name="Pets allowed",
        patterns=("pets allowed",),
        indexed=True,
        why=(
            "A hard requirement, and at 13.3% the clearest case for pushdown in "
            "the set: applied after a 100-candidate scan it leaves about 13 rows."
        ),
    ),
    AmenityGroup(
        column="has_air_conditioning",
        name="Air conditioning",
        patterns=(
            r"(?:central |portable |window )?air conditioning",
            r"window ac unit",
            r"ac - split type ductless system",
        ),
        indexed=True,
        why=(
            "Rare in London (11.8%) and asked for in summer, so the pool collapses "
            "without pushdown. Fans are deliberately not here: 'Portable fans' "
            "is in 19% of listings and merging it would answer a question about "
            "cooling with a desk fan."
        ),
    ),
    AmenityGroup(
        column="has_gym",
        name="Gym",
        patterns=(
            r"(?:private |shared )?gym(?: in building)?",
            r"exercise equipment(?::.*)?",
        ),
        indexed=True,
        why=(
            "The most selective group kept (7.7%), which is exactly why it needs "
            "the index: post-filtered it would leave a handful of rows. "
            "'Gym nearby' is excluded -- a gym the guest has to find and pay for "
            "is not the amenity being asked about."
        ),
    ),
    # --------------------------------------- considered, left out of the index --
    AmenityGroup(
        column="has_wifi",
        name="Wifi",
        patterns=(rf"(?:fast |pocket )?wi-?fi{_DETAIL}",),
        indexed=False,
        why=(
            "The most requested amenity there is, and at 97.1% it filters almost "
            "nothing: checking it against the array after the scan drops 3% of "
            "candidates. The speed-tested spellings ('Fast wifi - 73 Mbps') are "
            "folded in rather than kept as a 'fast wifi' group of their own -- "
            "the measurement exists for only 2% of listings, so a filter on it "
            "would rank listings by whether a host ran a speed test."
        ),
    ),
    AmenityGroup(
        column="has_kitchen",
        name="Kitchen",
        patterns=("kitchen", "kitchenette"),
        indexed=False,
        why=(
            "91.2% of listings, so a column buys next to nothing. 'Outdoor "
            "kitchen' is excluded: a guest asking for a kitchen means one they "
            "can cook in indoors."
        ),
    ),
    AmenityGroup(
        column="has_washer",
        name="Washer",
        patterns=(rf"(?:free |paid )?washer(?: - in (?:unit|building))?{_DETAIL}",),
        indexed=False,
        why=(
            "Asked for as often as the dryer but present in 85.9% of listings, "
            "and 99.6% of the listings with a dryer also have a washer -- so "
            "the pair's selectivity lives entirely in `has_dryer` and a request "
            "for both loses nothing. 'Laundromat nearby' is a different amenity "
            "and stays out."
        ),
    ),
    AmenityGroup(
        column="has_heating",
        name="Heating",
        patterns=(
            r"(?:central |radiant )?heating",
            r"heating - split type ductless system",
        ),
        indexed=False,
        why=(
            "85% of London listings, and the ones that omit it are mostly rooms "
            "whose host did not list it rather than cold flats. Nothing to "
            "filter and nobody asks."
        ),
    ),
    AmenityGroup(
        column="has_tv",
        name="TV",
        patterns=(r"(?:\d+ inch )?(?:hd)?tv(?: with .*)?",),
        indexed=False,
        why=(
            "76% once the 2,600-odd 'NN inch HDTV with <services>' spellings are "
            "folded in, and rarely the thing that decides a booking. The "
            "streaming services inside those strings are a stronger signal than "
            "the set itself, and they are left to the embedded document."
        ),
    ),
    AmenityGroup(
        column="has_long_term_stays",
        name="Long term stays allowed",
        patterns=("long term stays allowed",),
        indexed=False,
        why=(
            "Selective at 37.8%, but it means 'a stay of 28 days or more is "
            "permitted', which overlaps the `minimum_nights` column a filter can "
            "already express. Worth revisiting if a monthly-stay query class "
            "appears in the evaluation set."
        ),
    ),
    AmenityGroup(
        column="has_hot_tub",
        name="Hot tub",
        patterns=(r"(?:private |shared )?hot tub(?: - .*)?",),
        indexed=False,
        why=(
            "2.4% of listings, so post-filtering it empties the pool -- the "
            "strongest selectivity case in the data. It loses on demand: in a "
            "city snapshot it is a rounding error, and spending a permanent key "
            "slot on it would be optimizing for a query nobody types."
        ),
    ),
    AmenityGroup(
        column="has_free_street_parking",
        name="Free street parking",
        patterns=("free street parking",),
        indexed=False,
        why=(
            "Kept as its own group so the free-parking number stays honest about "
            "what it excludes. Not indexed: it describes the street rather than "
            "the listing, and a guest who needs a guaranteed space is not "
            "answered by it."
        ),
    ),
)

INDEXED_AMENITY_GROUPS: Final[tuple[AmenityGroup, ...]] = tuple(
    group for group in AMENITY_GROUPS if group.indexed
)

# PostgreSQL allows at most this many key columns in one index.
MAX_INDEX_KEY_COLUMNS: Final = 32

# The filter columns the index spends before any amenity, named rather than
# counted: a 17 written as a literal would still read as 17 after a migration
# added a filter column, and the amenity budget would silently be one too
# large. A schema test asserts this list is exactly the filterable set the
# database has.
FIXED_INDEX_FILTER_COLUMNS: Final[tuple[str, ...]] = (
    "neighbourhood_id",
    "room_type_id",
    "property_type_id",
    "price_gbp",
    "accommodates",
    "bedrooms",
    "beds",
    "bathrooms",
    "minimum_nights",
    "rating",
    "location_score",
    "number_of_reviews",
    "instant_bookable",
    "host_is_superhost",
    "latitude",
    "longitude",
)
# Plus one for the embedding, which is a key column like any other.
FIXED_INDEX_KEY_COLUMNS: Final = 1 + len(FIXED_INDEX_FILTER_COLUMNS)
AMENITY_COLUMN_BUDGET: Final = MAX_INDEX_KEY_COLUMNS - FIXED_INDEX_KEY_COLUMNS

INDEXED_AMENITY_COLUMNS: Final[tuple[str, ...]] = tuple(
    group.column for group in INDEXED_AMENITY_GROUPS
)


def indexed_amenity_columns() -> tuple[str, ...]:
    """The boolean columns, in schema order. One per indexed group."""
    return INDEXED_AMENITY_COLUMNS


@lru_cache(maxsize=_SPELLING_CACHE)
def groups_for_spelling(raw: str) -> tuple[AmenityGroup, ...]:
    """Every group that claims this raw amenity string, in group order.

    Usually empty -- most of the 8,735 spellings in a snapshot are a host's own
    words -- or exactly one. A tuple rather than a single value because nothing
    stops a later snapshot from publishing one string that means two amenities,
    and silently keeping the first match would be a wrong `false` with no
    symptom.
    """
    folded = fold_amenity(raw)
    return tuple(group for group in AMENITY_GROUPS if group.regex.fullmatch(folded) is not None)


# Cached as well as `groups_for_spelling`, because this is the answer the
# backfill's inner loop actually asks for: one listing names ~35 amenities and
# the snapshot repeats the same spellings across 92,000 of them.
@lru_cache(maxsize=_SPELLING_CACHE)
def columns_for_spelling(raw: str) -> frozenset[str]:
    """The indexed boolean columns one raw amenity string sets."""
    return frozenset(group.column for group in groups_for_spelling(raw) if group.indexed)


def columns_for_listing(amenities: Iterable[str]) -> frozenset[str]:
    """Which boolean columns are true for a listing with these amenities."""
    columns: set[str] = set()
    for amenity in amenities:
        columns |= columns_for_spelling(amenity)
    return frozenset(columns)


def booleans_for_listing(amenities: Iterable[str]) -> tuple[bool, ...]:
    """The indexed booleans for one listing, in `indexed_amenity_columns()` order."""
    present = columns_for_listing(amenities)
    return tuple(column in present for column in INDEXED_AMENITY_COLUMNS)


@dataclass(frozen=True, slots=True)
class GroupCount:
    """How prevalent one group is in the data that was counted."""

    group: AmenityGroup
    listings: int
    spellings: int

    def share(self, total_listings: int) -> float:
        """Fraction of listings carrying this group, or 0.0 when there are none."""
        return self.listings / total_listings if total_listings else 0.0


@dataclass(frozen=True, slots=True)
class AmenityReport:
    """Group prevalence over a corpus, plus what no group claimed.

    `uncovered` is the honest half of the report: the most common spellings that
    fall outside every group, which is the only way a reader can tell a
    deliberate omission from a pattern that quietly stopped matching.
    """

    listings: int
    counts: tuple[GroupCount, ...]
    uncovered: tuple[tuple[str, int], ...]
    distinct_spellings: int


def count_amenity_groups(
    listing_amenities: Iterable[Sequence[str]], *, uncovered_limit: int = 15
) -> AmenityReport:
    """Count how many listings carry each group. Pure: no database, no settings.

    Args:
        listing_amenities: one amenity list per listing.
        uncovered_limit: how many unclaimed spellings to keep, by listing count.

    Returns:
        Counts ranked by prevalence, highest first, with ties broken by group
        order so the same corpus always produces the same report.
    """
    listings = 0
    per_group: Counter[str] = Counter()
    spellings_per_group: dict[str, set[str]] = {group.column: set() for group in AMENITY_GROUPS}
    uncovered: Counter[str] = Counter()
    all_spellings: set[str] = set()

    for amenities in listing_amenities:
        listings += 1
        matched_here: set[str] = set()
        for amenity in amenities:
            all_spellings.add(amenity)
            claiming = groups_for_spelling(amenity)
            if not claiming:
                uncovered[amenity] += 1
            for group in claiming:
                matched_here.add(group.column)
                spellings_per_group[group.column].add(amenity)
        per_group.update(matched_here)

    counts = tuple(
        sorted(
            (
                GroupCount(
                    group=group,
                    listings=per_group[group.column],
                    spellings=len(spellings_per_group[group.column]),
                )
                for group in AMENITY_GROUPS
            ),
            key=lambda count: (-count.listings, AMENITY_GROUPS.index(count.group)),
        )
    )
    return AmenityReport(
        listings=listings,
        counts=counts,
        uncovered=tuple(uncovered.most_common(uncovered_limit)),
        distinct_spellings=len(all_spellings),
    )


def render_report(report: AmenityReport) -> str:
    """The ranked table, the recommendation, and the index-key arithmetic.

    Pure, so the exact text a reviewer reads is what a test asserts on.
    """
    lines = [
        f"{report.listings:,} listings, {report.distinct_spellings:,} distinct "
        f"amenity spellings, {len(AMENITY_GROUPS)} groups",
        "",
        f"  {'share':>7}  {'listings':>9}  {'spellings':>9}  {'indexed':<7}  group / column",
    ]
    for count in report.counts:
        indexed = "column" if count.group.indexed else "array"
        lines.append(
            f"  {count.share(report.listings) * 100:6.2f}%  {count.listings:9,}  "
            f"{count.spellings:9,}  {indexed:<7}  {count.group.name} "
            f"({count.group.column})"
        )

    lines += [
        "",
        f"index keys: {FIXED_INDEX_KEY_COLUMNS} fixed (embedding + 16 filter "
        f"columns) + {len(INDEXED_AMENITY_GROUPS)} amenity booleans = "
        f"{FIXED_INDEX_KEY_COLUMNS + len(INDEXED_AMENITY_GROUPS)} of "
        f"{MAX_INDEX_KEY_COLUMNS} allowed "
        f"({AMENITY_COLUMN_BUDGET - len(INDEXED_AMENITY_GROUPS)} amenity slots "
        f"unspent)",
        "",
        "indexed, and why:",
    ]
    for group in INDEXED_AMENITY_GROUPS:
        lines.append(f"  {group.column:<26} {group.why}")

    lines += ["", "considered and left in the array:"]
    for group in AMENITY_GROUPS:
        if not group.indexed:
            lines.append(f"  {group.column:<26} {group.why}")

    lines += [
        "",
        "most common spellings no group claims (a filter on one of these is an array post-filter):",
    ]
    for spelling, listings in report.uncovered:
        share = listings / report.listings * 100 if report.listings else 0.0
        lines.append(f"  {share:6.2f}%  {listings:9,}  {spelling}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The database side: reading the corpus, and writing the decision into it.
# --------------------------------------------------------------------------- #


class AmenityError(RuntimeError):
    """The amenity report or backfill could not run against this database.

    Separate from a bad amenity string, which is data and is counted: this one
    means the schema is not the one these groups were written for -- most often
    a database that has not had the amenity-column migration applied.
    """


# `cursor.stream` rather than `execute`, so rows arrive as they are read. A
# plain psycopg cursor transfers the whole result set when `execute` returns,
# and one snapshot's amenity arrays are ~3.2 million strings: ~320 MB held in
# Python, inside the load's open transaction, to compute one bool per column.
# Streaming them instead costs a few seconds and holds 43 MB -- the listing ids
# and the booleans themselves, which the update statement needs all at once.
# A server-side cursor would do the same job but has to be DECLAREd, which an
# autocommit connection cannot do, and what kind of connection a caller hands
# over is none of this module's business.
_READ_LISTING_AMENITIES: Final = "SELECT amenities FROM listings"
_READ_LISTING_IDS_AND_AMENITIES: Final = "SELECT id, amenities FROM listings ORDER BY id"
_READ_LOOKUP_NAMES: Final = "SELECT id, name FROM amenities"

# Recomputed from the array rather than nudged, because the booleans are derived
# data: a listing whose amenity list changed must not keep a flag from the list
# it used to have. The values arrive as parallel arrays -- one listing id array
# and one boolean array per column -- so the whole table is one parameterized
# statement with no identifier interpolated by hand.
#
# The row comparison matters as much as the assignment. Without it every
# backfill rewrites all 92,000 rows whether or not a flag moved, which on a
# reload is a second full-table write inside the same transaction as the upsert
# that already rewrote it. With it, a no-op backfill writes nothing and says so.
_UPDATE_LISTING_BOOLEANS: Final = """
UPDATE listings SET {assignments}
FROM (SELECT * FROM unnest({ids_and_flags})) AS derived (id, {columns})
WHERE listings.id = derived.id
  AND ({current}) IS DISTINCT FROM ({proposed})
"""

_UPDATE_LOOKUP_COLUMNS: Final = """
UPDATE amenities SET indexed_column = derived.indexed_column
FROM (SELECT * FROM unnest(%s::int4[], %s::text[])) AS derived (id, indexed_column)
WHERE amenities.id = derived.id
  AND amenities.indexed_column IS DISTINCT FROM derived.indexed_column
"""

# Reading zero rows, purely to find out whether the columns exist. A backfill
# over an empty table would otherwise touch none of them and report success on a
# schema that cannot hold the answer -- the failure mode this project refuses,
# because "nothing to derive" and "the schema is wrong" would look identical.
_PROBE_AMENITY_SCHEMA: Final = """
SELECT {listing_columns} FROM listings LIMIT 0
"""


def _require_amenity_schema(cur: psycopg.Cursor[TupleRow]) -> None:
    """Fail now, loudly, if the schema has no amenity columns to derive into.

    Raises:
        AmenityError: naming the migration to apply.
    """
    try:
        cur.execute(
            sql.SQL(_PROBE_AMENITY_SCHEMA).format(
                listing_columns=sql.SQL(", ").join(
                    sql.Identifier(column) for column in INDEXED_AMENITY_COLUMNS
                )
            )
        )
        cur.execute("SELECT indexed_column FROM amenities LIMIT 0")
    except psycopg.errors.UndefinedColumn as exc:
        raise AmenityError(
            f"this schema has no amenity columns to derive into ({exc}); apply the migrations first"
        ) from exc
    except psycopg.errors.UndefinedTable as exc:
        raise AmenityError(
            f"this schema has no listings or amenities table ({exc}); apply the migrations first"
        ) from exc


def read_amenity_report(conn: Connection, *, uncovered_limit: int = 15) -> AmenityReport:
    """Count the groups over every loaded listing.

    Reads nothing but the amenity arrays, and streams them, so the count does
    not depend on the corpus fitting in memory.

    Raises:
        AmenityError: if the listings table cannot be read.
    """
    try:
        with conn.cursor() as cur:
            return count_amenity_groups(
                (row[0] or () for row in cur.stream(_READ_LISTING_AMENITIES)),
                uncovered_limit=uncovered_limit,
            )
    except psycopg.Error as exc:
        raise AmenityError(f"could not read listing amenities: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """What one backfill derived, and how much of it was new."""

    listings: int
    columns: tuple[str, ...]
    per_column: Mapping[str, int]
    listings_changed: int
    lookup_rows_flagged: int
    lookup_rows_changed: int


def backfill_amenity_columns(conn: Connection) -> BackfillReport:
    """Derive every indexed boolean from `listings.amenities`, and flag the lookup.

    Runs in the caller's transaction, so a load can make the booleans part of
    the same all-or-nothing write that produced the arrays they are derived from.
    Only rows whose flags actually change are written, so running it twice is a
    no-op rather than a second full-table rewrite.

    The matching stays in Python even though the update is SQL: a second copy of
    the patterns in SQL would be a second thing to keep in step with the group
    table, and the symptom of drift is a listing that quietly stops matching an
    amenity it has.

    Raises:
        AmenityError: if the boolean columns or `amenities.indexed_column` are
            missing -- which means the schema predates this amenity set -- or if
            the derivation itself fails. An empty database is not an error, but
            it is also not allowed to hide a missing column: the schema is
            checked before any row is read.
    """
    columns = INDEXED_AMENITY_COLUMNS
    try:
        with conn.cursor() as cur:
            _require_amenity_schema(cur)

        ids: list[int] = []
        flags: list[list[bool]] = [[] for _ in columns]
        per_column: Counter[str] = Counter()
        with conn.cursor() as reader:
            for listing_id, amenities in reader.stream(_READ_LISTING_IDS_AND_AMENITIES):
                ids.append(int(listing_id))
                for index, value in enumerate(booleans_for_listing(amenities or ())):
                    flags[index].append(value)
                    if value:
                        per_column[columns[index]] += 1

        with conn.cursor() as cur:
            listings_changed = 0
            if ids:
                cur.execute(_listing_update_statement(columns), (ids, *flags))
                listings_changed = cur.rowcount

            lookup_ids: list[int] = []
            lookup_columns: list[str | None] = []
            flagged = 0
            cur.execute(_READ_LOOKUP_NAMES)
            for amenity_id, name in cur.fetchall():
                matched = columns_for_spelling(str(name))
                # Several columns for one spelling is possible by construction;
                # the lookup table holds one, so the decision has to be stable
                # rather than whichever the set iterated first.
                column = min(matched) if matched else None
                lookup_ids.append(int(amenity_id))
                lookup_columns.append(column)
                if column is not None:
                    flagged += 1
            lookup_changed = 0
            if lookup_ids:
                cur.execute(_UPDATE_LOOKUP_COLUMNS, (lookup_ids, lookup_columns))
                lookup_changed = cur.rowcount
    except psycopg.Error as exc:
        raise AmenityError(f"could not derive the amenity columns: {exc}") from exc

    return BackfillReport(
        listings=len(ids),
        columns=columns,
        per_column=dict(per_column),
        listings_changed=listings_changed,
        lookup_rows_flagged=flagged,
        lookup_rows_changed=lookup_changed,
    )


def _listing_update_statement(columns: Sequence[str]) -> sql.Composed:
    """`UPDATE listings SET has_x = derived.has_x, ...` for the indexed columns.

    Identifiers are composed, never formatted into a string, even though they
    come from this module rather than from input.
    """
    identifiers = [sql.Identifier(column) for column in columns]
    return sql.SQL(_UPDATE_LISTING_BOOLEANS).format(
        assignments=sql.SQL(", ").join(
            sql.SQL("{column} = derived.{column}").format(column=identifier)
            for identifier in identifiers
        ),
        ids_and_flags=sql.SQL(", ").join(
            [sql.SQL("%s::int4[]"), *(sql.SQL("%s::bool[]") for _ in columns)]
        ),
        columns=sql.SQL(", ").join(identifiers),
        current=sql.SQL(", ").join(
            sql.SQL("listings.{column}").format(column=identifier) for identifier in identifiers
        ),
        proposed=sql.SQL(", ").join(
            sql.SQL("derived.{column}").format(column=identifier) for identifier in identifiers
        ),
    )
