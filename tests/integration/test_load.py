"""What a load actually leaves in the database.

Run against a synthetic snapshot, never real listings: the fixture is shaped
like the London file -- the same columns in the same order, including the
personal ones the parsers are supposed to drop -- but every row in it is made
up, which is what lets it live in the repository at all.

The migrations are applied into a throwaway schema, so the sandbox's own
database is left alone and the test can be run again.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from psycopg.rows import TupleRow

from scout.config import get_settings
from scout.data.load import LoadError, LoadReport, load_into
from scout.data.migrate import apply_migrations
from scout.data.parsers import MIN_PROPERTY_TYPE_LISTINGS, OTHER_PROPERTY_TYPE

pytestmark = pytest.mark.integration

TEST_SCHEMA = "scout_load_test"

SNAPSHOT_DATE = date(2026, 6, 19)
MAX_REVIEWS = 2

# The columns the snapshot ships, in its order. The personal ones are here on
# purpose: a loader can only be shown dropping them if they were present.
LISTING_HEADER = (
    "id",
    "listing_url",
    "name",
    "description",
    "host_id",
    "host_url",
    "host_name",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_is_superhost",
    "neighbourhood_cleansed",
    "latitude",
    "longitude",
    "property_type",
    "room_type",
    "accommodates",
    "bathrooms",
    "bathrooms_text",
    "bedrooms",
    "beds",
    "amenities",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "review_scores_rating",
    "review_scores_location",
    "review_scores_cleanliness",
    "instant_bookable",
)

REVIEW_HEADER = ("listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments")

# One distinctive string in every personal column, so a test can look for it in
# the database rather than asking whether a column exists. Neither appears in
# any column Scout keeps, which is what makes them worth searching for.
HOST_NAME = "Zephyrine"
REVIEWER_NAME = "Quillon"
# A name written into the body of a review, which is a different problem from a
# name in a column: guests do it constantly, and removing it is the scrubber's
# job, later in the pipeline. Stored raw here, deliberately.
NAME_INSIDE_A_REVIEW = "Marisol"

CANAL = 1
PARK_ROOM = 2
LIGHTHOUSE = 3
UNUSABLE = 4
ABSENT_FROM_LISTINGS = 999
# Enough listings of one type to clear the rare-type threshold, so the
# collapsing under test is the production rule rather than a test-only one.
FILLER_IDS = tuple(range(100, 100 + MIN_PROPERTY_TYPE_LISTINGS + 5))


def listing(source_id: int, **overrides: str) -> dict[str, str]:
    """One row in the snapshot's shape, with sensible London-ish defaults."""
    row = {
        "id": str(source_id),
        "listing_url": f"https://www.airbnb.com/rooms/{source_id}",
        "name": f"Listing {source_id}",
        "description": "A place to stay.",
        "host_id": str(500_000 + source_id),
        "host_url": f"https://www.airbnb.com/users/show/{500_000 + source_id}",
        "host_name": HOST_NAME,
        "host_about": f"{HOST_NAME} has hosted for years.",
        "host_thumbnail_url": "https://example.invalid/thumb.jpg",
        "host_picture_url": "https://example.invalid/pic.jpg",
        "host_is_superhost": "f",
        "neighbourhood_cleansed": "Hackney",
        "latitude": "51.5450",
        "longitude": "-0.0553",
        "property_type": "Entire rental unit",
        "room_type": "Entire home/apt",
        "accommodates": "2",
        "bathrooms": "1.0",
        "bathrooms_text": "1 bath",
        "bedrooms": "1",
        "beds": "1",
        "amenities": '["Wifi", "Kitchen"]',
        "price": "$150.00",
        "minimum_nights": "2",
        "number_of_reviews": "0",
        "review_scores_rating": "4.80",
        "review_scores_location": "4.70",
        "review_scores_cleanliness": "4.90",
        # Empty for every row of the real snapshot, which is the case that
        # forced the column to become nullable.
        "instant_bookable": "",
    }
    return row | overrides


def review(source_listing_id: int, when: str, comments: str) -> dict[str, str]:
    return {
        "listing_id": str(source_listing_id),
        "id": str(abs(hash((source_listing_id, when))) % 10_000_000),
        "date": when,
        "reviewer_id": "873421",
        "reviewer_name": REVIEWER_NAME,
        "comments": comments,
    }


def default_listings() -> list[dict[str, str]]:
    return [
        listing(
            CANAL,
            name="Canalside studio",
            description="Quiet, two minutes from the water.",
            host_is_superhost="t",
            amenities='["Wifi", "Kitchen", "Dedicated workspace"]',
            price="$1,234.00",
            number_of_reviews="4",
        ),
        listing(
            PARK_ROOM,
            name="Bright room by the park",
            neighbourhood_cleansed="Islington",
            room_type="Private room",
            property_type="Private room in home",
            bathrooms_text="1 shared bath",
            # The same amenity, written the way a minority of listings write it.
            amenities='["wifi", "Iron"]',
            # A third of London's listings carry no price at all.
            price="",
            number_of_reviews="1",
        ),
        listing(
            LIGHTHOUSE,
            name="A night in a lighthouse",
            property_type="Lighthouse",
            bathrooms_text="Shared half-bath",
            price="$300.00",
        ),
        # NOT NULL in the schema, missing in the file: five real London listings
        # look like this, and half a listing is not stored.
        listing(UNUSABLE, name="Missing its minimum stay", minimum_nights=""),
        *(listing(source_id) for source_id in FILLER_IDS),
    ]


def default_reviews() -> list[dict[str, str]]:
    return [
        review(CANAL, "2024-01-01", "Oldest, and should not survive the cap."),
        review(CANAL, "2025-01-01", "Older, and should not survive the cap either."),
        review(CANAL, "2025-06-01", "Second most recent."),
        review(CANAL, "2025-06-02", "Most recent."),
        review(PARK_ROOM, "2025-05-05", f"{NAME_INSIDE_A_REVIEW} was a wonderful host."),
        review(UNUSABLE, "2025-05-05", "For a listing that could not be stored."),
        review(ABSENT_FROM_LISTINGS, "2025-05-06", "For a listing not in this snapshot."),
        review(ABSENT_FROM_LISTINGS, "2025-05-07", "Also for a listing not in this snapshot."),
    ]


@dataclass(frozen=True, slots=True)
class Snapshot:
    """A pair of gzipped CSVs on disk, rewritable between loads."""

    listings_csv: Path
    reviews_csv: Path

    def write(
        self, listings: Sequence[dict[str, str]], reviews: Sequence[dict[str, str]]
    ) -> Snapshot:
        _write_csv(self.listings_csv, LISTING_HEADER, listings)
        _write_csv(self.reviews_csv, REVIEW_HEADER, reviews)
        return self


def _write_csv(path: Path, header: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def conn() -> Iterator[psycopg.Connection[TupleRow]]:
    dsn = get_settings().database_url
    try:
        connection = psycopg.connect(dsn, connect_timeout=5, autocommit=True)
    except psycopg.Error as exc:
        pytest.skip(f"database not reachable: {exc}")
    with connection:
        yield connection


@pytest.fixture(scope="module")
def schema(conn: psycopg.Connection[TupleRow]) -> Iterator[None]:
    """Scout's schema, applied fresh into a namespace this test owns."""
    drop = sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(TEST_SCHEMA))
    conn.execute(drop)
    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(TEST_SCHEMA)))
    conn.execute(sql.SQL("SET search_path TO {}, public").format(sql.Identifier(TEST_SCHEMA)))
    apply_migrations(conn)
    yield
    conn.execute(drop)


@pytest.fixture
def snapshot(tmp_path: Path) -> Snapshot:
    return Snapshot(tmp_path / "listings.csv.gz", tmp_path / "reviews.csv.gz")


def load(
    conn: psycopg.Connection[TupleRow], snapshot: Snapshot, *, max_reviews: int = MAX_REVIEWS
) -> LoadReport:
    return load_into(
        conn,
        listings_csv=snapshot.listings_csv,
        reviews_csv=snapshot.reviews_csv,
        snapshot_date=SNAPSHOT_DATE,
        city="London",
        max_reviews_per_listing=max_reviews,
    )


@pytest.fixture
def loaded(
    conn: psycopg.Connection[TupleRow], schema: None, snapshot: Snapshot
) -> tuple[LoadReport, Snapshot]:
    snapshot.write(default_listings(), default_reviews())
    return load(conn, snapshot), snapshot


def scalar(conn: psycopg.Connection[TupleRow], query: str, *args: object) -> object:
    row = conn.execute(query, args or None).fetchone()
    assert row is not None
    return row[0]


class TestWhatOneLoadWrites:
    def test_listings_reviews_and_every_lookup_are_populated(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        report, _ = loaded
        expected_listings = len(default_listings()) - 1  # the unusable one

        assert report.listings_seen == len(default_listings())
        assert report.listings_loaded == expected_listings
        assert scalar(conn, "SELECT count(*) FROM listings") == expected_listings
        assert scalar(conn, "SELECT count(*) FROM neighbourhoods") == 2
        assert scalar(conn, "SELECT count(*) FROM room_types") == 2
        assert scalar(conn, "SELECT count(*) FROM amenities") >= 4
        assert scalar(conn, "SELECT count(*) FROM reviews") == report.reviews_kept

    def test_a_listings_fields_survive_the_round_trip(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        row = conn.execute(
            "SELECT l.name, n.name, r.name, p.name, l.price_gbp, l.accommodates, "
            "l.bathrooms, l.rating, l.host_is_superhost, l.amenities "
            "FROM listings l "
            "JOIN neighbourhoods n ON n.id = l.neighbourhood_id "
            "JOIN room_types r ON r.id = l.room_type_id "
            "JOIN property_types p ON p.id = l.property_type_id "
            "WHERE l.source_listing_id = %s",
            (CANAL,),
        ).fetchone()

        assert row == (
            "Canalside studio",
            "Hackney",
            "Entire home/apt",
            "Entire rental unit",
            pytest.approx(1234.00),
            2,
            pytest.approx(1.0),
            pytest.approx(4.80),
            True,
            ["Wifi", "Kitchen", "Dedicated workspace"],
        )

    def test_a_listing_with_no_price_is_null_rather_than_free(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        assert (
            scalar(conn, "SELECT price_gbp FROM listings WHERE source_listing_id = %s", PARK_ROOM)
            is None
        )
        # And the NULL behaves the way the schema's comment says it does.
        assert scalar(conn, "SELECT count(*) FROM listings WHERE price_gbp < 1000000") == (
            len(default_listings()) - 2
        )

    def test_instant_bookable_is_unknown_rather_than_false(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """The snapshot stopped publishing the column; NULL is what it now says."""
        assert scalar(conn, "SELECT count(*) FROM listings WHERE instant_bookable IS NOT NULL") == 0

    def test_a_half_bath_is_half_a_bathroom(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        assert scalar(
            conn, "SELECT bathrooms FROM listings WHERE source_listing_id = %s", LIGHTHOUSE
        ) == pytest.approx(0.5)

    def test_a_rare_property_type_is_collapsed(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """One lighthouse cannot support a filter, and costs a line in the prompt."""
        report, _ = loaded
        assert report.property_types_collapsed == 2

        collapsed = scalar(
            conn,
            "SELECT p.name FROM listings l JOIN property_types p ON p.id = l.property_type_id "
            "WHERE l.source_listing_id = %s",
            LIGHTHOUSE,
        )
        assert collapsed == OTHER_PROPERTY_TYPE
        assert scalar(conn, "SELECT count(*) FROM property_types WHERE name = 'Lighthouse'") == 0

    def test_one_amenity_written_two_ways_is_one_amenity(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Splitting an amenity across spellings would also split its count.

        The indexed amenity selection is ranked on those counts, so the more
        common spelling wins for both listings.
        """
        assert scalar(conn, "SELECT count(*) FROM amenities WHERE lower(name) = 'wifi'") == 1
        assert scalar(
            conn, "SELECT amenities FROM listings WHERE source_listing_id = %s", PARK_ROOM
        ) == ["Wifi", "Iron"]

    def test_the_snapshot_this_data_came_from_is_recorded(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Listing ids are not stable between releases; a number needs its date."""
        report, _ = loaded
        row = conn.execute(
            "SELECT snapshot_date, city, listing_count, review_count FROM data_snapshot"
        ).fetchone()
        assert row == (SNAPSHOT_DATE, "London", report.listings_loaded, report.reviews_kept)


class TestReviews:
    def test_only_the_most_recent_reviews_per_listing_are_kept(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        kept = conn.execute(
            "SELECT r.date, r.comments FROM reviews r JOIN listings l ON l.id = r.listing_id "
            "WHERE l.source_listing_id = %s ORDER BY r.date DESC",
            (CANAL,),
        ).fetchall()

        assert [row[0] for row in kept] == [date(2025, 6, 2), date(2025, 6, 1)]
        assert all("should not survive" not in row[1] for row in kept)

    def test_no_listing_exceeds_the_cap(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        most = scalar(
            conn,
            "SELECT coalesce(max(per_listing), 0) FROM "
            "(SELECT count(*) AS per_listing FROM reviews GROUP BY listing_id) counts",
        )
        assert isinstance(most, int)
        assert most <= MAX_REVIEWS

    def test_reviews_for_a_listing_outside_the_snapshot_are_counted_not_stored(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """London's file carries a few thousand; a reported number is checkable."""
        report, _ = loaded
        # Two for a listing that is not in the file, one for a rejected listing.
        assert report.reviews_without_listing == 3
        assert report.reviews_seen == len(default_reviews())
        assert scalar(conn, "SELECT count(*) FROM reviews") == 3

    def test_a_listing_leaving_the_snapshot_takes_its_reviews_with_it(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Reviews exist to cite a listing; an orphan is one nothing can show."""
        _, snapshot = loaded
        snapshot.write(
            [row for row in default_listings() if row["id"] != str(CANAL)], default_reviews()
        )

        load(conn, snapshot)

        assert (
            scalar(conn, "SELECT count(*) FROM listings WHERE source_listing_id = %s", CANAL) == 0
        )
        # Only the room's single review is left; the studio's two are gone.
        assert scalar(conn, "SELECT count(*) FROM reviews") == 1


class TestRejectedRows:
    def test_a_listing_missing_a_not_null_column_is_rejected_and_reported(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        report, _ = loaded

        assert report.listings_rejected == 1
        assert report.listing_rejections == (("minimum_nights", 1),)
        assert (
            scalar(conn, "SELECT count(*) FROM listings WHERE source_listing_id = %s", UNUSABLE)
            == 0
        )

    def test_a_file_of_nothing_but_bad_rows_does_not_empty_the_database(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """A failure is never an empty result. The previous load must survive it."""
        report, snapshot = loaded
        snapshot.write([listing(source_id, minimum_nights="") for source_id in FILLER_IDS], [])

        with pytest.raises(LoadError) as caught:
            load(conn, snapshot)

        assert "minimum_nights" in str(caught.value)
        assert scalar(conn, "SELECT count(*) FROM listings") == report.listings_loaded

    def test_a_renamed_column_stops_the_load_rather_than_rejecting_every_row(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        report, snapshot = loaded
        renamed = [
            {key: value for key, value in row.items() if key != "price"} | {"cost": "$1"}
            for row in default_listings()
        ]
        _write_csv(
            snapshot.listings_csv,
            (*(column for column in LISTING_HEADER if column != "price"), "cost"),
            renamed,
        )

        with pytest.raises(LoadError) as caught:
            load(conn, snapshot)

        assert "price" in str(caught.value)
        assert scalar(conn, "SELECT count(*) FROM listings") == report.listings_loaded

    def test_a_missing_file_says_so(
        self, conn: psycopg.Connection[TupleRow], schema: None, tmp_path: Path
    ) -> None:
        missing = Snapshot(tmp_path / "nowhere.csv.gz", tmp_path / "also-nowhere.csv.gz")
        with pytest.raises(LoadError) as caught:
            load(conn, missing)
        assert "nowhere.csv.gz" in str(caught.value)


class TestReloading:
    def test_reloading_the_same_snapshot_changes_nothing(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        first, snapshot = loaded
        before = conn.execute(
            "SELECT id, source_listing_id FROM listings ORDER BY source_listing_id"
        ).fetchall()

        second = load(conn, snapshot)

        assert second.listings_loaded == first.listings_loaded
        assert second.reviews_kept == first.reviews_kept
        assert second.listings_removed == 0
        # The generated ids are stable: reviews and, later, relevance labels
        # reference them.
        assert (
            conn.execute(
                "SELECT id, source_listing_id FROM listings ORDER BY source_listing_id"
            ).fetchall()
            == before
        )
        assert scalar(conn, "SELECT count(*) FROM data_snapshot") == 1

    def test_a_changed_listing_is_updated_in_place(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        _, snapshot = loaded
        listing_id = scalar(conn, "SELECT id FROM listings WHERE source_listing_id = %s", PARK_ROOM)

        updated = [
            row | ({"price": "$95.00"} if row["id"] == str(PARK_ROOM) else {})
            for row in default_listings()
        ]
        snapshot.write(updated, default_reviews())
        load(conn, snapshot)

        assert scalar(
            conn, "SELECT price_gbp FROM listings WHERE source_listing_id = %s", PARK_ROOM
        ) == pytest.approx(95.0)
        assert scalar(conn, "SELECT id FROM listings WHERE source_listing_id = %s", PARK_ROOM) == (
            listing_id
        )

    def test_a_listing_that_left_the_snapshot_is_removed(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """The database says what the file says, or the counts stop meaning anything."""
        first, snapshot = loaded
        snapshot.write(
            [row for row in default_listings() if row["id"] != str(LIGHTHOUSE)], default_reviews()
        )

        second = load(conn, snapshot)

        assert second.listings_removed == 1
        assert second.listings_loaded == first.listings_loaded - 1
        assert (
            scalar(conn, "SELECT count(*) FROM listings WHERE source_listing_id = %s", LIGHTHOUSE)
            == 0
        )

    def test_an_embedding_survives_a_reload_that_changed_nothing(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Embedding is the expensive step; an idempotent reload must not redo it."""
        _, snapshot = loaded
        conn.execute(
            "UPDATE listings SET doc_text = %s, embedding = %s::real[]::brindle_vector "
            "WHERE source_listing_id = %s",
            ("Canalside studio. Quiet, two minutes from the water.", [0.1, 0.2, 0.3], CANAL),
        )

        load(conn, snapshot)

        assert scalar(
            conn,
            "SELECT embedding IS NOT NULL FROM listings WHERE source_listing_id = %s",
            CANAL,
        )

    def test_an_embedding_is_dropped_when_the_document_it_was_built_from_changed(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """A stale embedding is worse than none: nothing downstream can tell."""
        _, snapshot = loaded
        conn.execute(
            "UPDATE listings SET doc_text = %s, embedding = %s::real[]::brindle_vector "
            "WHERE source_listing_id = %s",
            ("Canalside studio. Quiet, two minutes from the water.", [0.1, 0.2, 0.3], CANAL),
        )

        renamed = [
            row | ({"name": "Canalside studio, refurbished"} if row["id"] == str(CANAL) else {})
            for row in default_listings()
        ]
        snapshot.write(renamed, default_reviews())
        load(conn, snapshot)

        row = conn.execute(
            "SELECT doc_text, embedding IS NULL FROM listings WHERE source_listing_id = %s",
            (CANAL,),
        ).fetchone()
        assert row == (None, True)

    def test_an_indexed_amenity_stays_indexed_across_a_reload(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """is_indexed is the amenity selection's answer, not the loader's to reset."""
        _, snapshot = loaded
        conn.execute("UPDATE amenities SET is_indexed = true WHERE name = 'Wifi'")

        load(conn, snapshot)

        assert scalar(conn, "SELECT is_indexed FROM amenities WHERE name = 'Wifi'") is True


class TestSnapshotsThatWouldOtherwiseKillTheLoad:
    """Shapes London does not have, which a reload or another city may.

    Each of these used to take the whole load down over a single row.
    """

    def test_a_listing_id_that_repeats_does_not_abort_the_load(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """The first occurrence wins, and the repeat is counted."""
        first, snapshot = loaded
        listings = default_listings()
        repeat = listing(CANAL, name="Canalside studio, listed twice", price="$99.00")
        snapshot.write([*listings, repeat], default_reviews())

        second = load(conn, snapshot)

        assert second.listings_duplicated == 1
        assert second.listings_loaded == first.listings_loaded
        assert scalar(conn, "SELECT name FROM listings WHERE source_listing_id = %s", CANAL) == (
            "Canalside studio"
        )

    def test_a_listing_that_became_unusable_keeps_its_row(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Rejected is not the same as departed.

        Its generated id is referenced by reviews and, later, by relevance
        labels; dropping it because one field went blank would cost more than
        the stale attribute it saves.
        """
        first, snapshot = loaded
        listing_id = scalar(conn, "SELECT id FROM listings WHERE source_listing_id = %s", PARK_ROOM)

        broken = [
            row | ({"minimum_nights": ""} if row["id"] == str(PARK_ROOM) else {})
            for row in default_listings()
        ]
        snapshot.write(broken, default_reviews())
        second = load(conn, snapshot)

        assert second.listings_rejected == 2
        assert second.listings_removed == 0
        assert scalar(conn, "SELECT id FROM listings WHERE source_listing_id = %s", PARK_ROOM) == (
            listing_id
        )
        assert second.listings_loaded == first.listings_loaded - 1

    def test_a_value_no_column_can_hold_costs_one_row_not_the_load(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """A NUL byte and an out-of-range integer both used to abort the COPY."""
        first, snapshot = loaded
        listings = [
            *default_listings(),
            listing(5001, name="Studio with a \x00 in its name"),
            listing(5002, bedrooms="40000"),
        ]
        snapshot.write(listings, default_reviews())

        second = load(conn, snapshot)

        assert second.listings_loaded == first.listings_loaded
        assert second.listings_rejected == 3
        assert dict(second.listing_rejections) == {"minimum_nights": 1, "name": 1, "bedrooms": 1}

    def test_a_truncated_download_is_a_load_error_not_a_traceback(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Half a gzip raises EOFError, which is not an OSError."""
        report, snapshot = loaded
        whole = snapshot.reviews_csv.read_bytes()
        snapshot.reviews_csv.write_bytes(whole[: len(whole) // 2])

        with pytest.raises(LoadError) as caught:
            load(conn, snapshot)

        assert str(snapshot.reviews_csv) in str(caught.value)
        assert scalar(conn, "SELECT count(*) FROM listings") == report.listings_loaded
        assert scalar(conn, "SELECT count(*) FROM reviews") == report.reviews_kept


class TestPrivacy:
    """docs/DATA.md section 4, checked against the values rather than the columns.

    Every fixture row carries a host name, a host id, and a reviewer name. A
    column-name check would miss a loader that wrote a name into `description`;
    this looks for the strings themselves.
    """

    @pytest.mark.parametrize("name", [HOST_NAME, REVIEWER_NAME, str(500_000 + CANAL)])
    def test_no_personal_value_from_the_file_reaches_any_table(
        self,
        conn: psycopg.Connection[TupleRow],
        loaded: tuple[LoadReport, Snapshot],
        name: str,
    ) -> None:
        pattern = f"%{name}%"
        in_listings = scalar(
            conn,
            "SELECT count(*) FROM listings WHERE name LIKE %s "
            "OR coalesce(description, '') LIKE %s "
            "OR array_to_string(amenities, ' ') LIKE %s "
            "OR coalesce(doc_text, '') LIKE %s",
            pattern,
            pattern,
            pattern,
            pattern,
        )
        assert in_listings == 0, f"{name} reached listings"
        assert scalar(conn, "SELECT count(*) FROM reviews WHERE comments LIKE %s", pattern) == 0, (
            f"{name} reached reviews"
        )

    def test_a_name_inside_a_review_body_is_still_there(
        self, conn: psycopg.Connection[TupleRow], loaded: tuple[LoadReport, Snapshot]
    ) -> None:
        """Stored raw on purpose, and this test exists so that stays deliberate.

        Guests write names into reviews constantly. Removing them is one tested
        scrubbing function that runs before the text is embedded or shown, not
        something this load does quietly on its way past -- a review rewritten
        here could never be checked against the source again.
        """
        stored = scalar(
            conn,
            "SELECT count(*) FROM reviews WHERE comments LIKE %s",
            f"%{NAME_INSIDE_A_REVIEW}%",
        )
        assert stored == 1
