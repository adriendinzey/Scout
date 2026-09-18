"""The loader's file handling, which needs no database to be wrong.

The streaming test is the one worth reading: the review file is 2.2 million
rows, and "it worked on my fixture" is exactly what a loader that quietly reads
the whole thing into memory would also say.
"""

from __future__ import annotations

import gzip
import tracemalloc
from datetime import date
from pathlib import Path

import pytest

from scout.data.load import (
    LoadError,
    _dict_rows,
    _parsed_listings,
    _parsed_reviews,
    _require_snapshot_date,
    _RowStats,
)
from scout.data.parsers import REQUIRED_REVIEW_COLUMNS

REVIEW_HEADER = "listing_id,id,date,reviewer_id,reviewer_name,comments\n"


def write_reviews(path: Path, count: int) -> Path:
    """A gzipped review file with `count` rows of realistic width."""
    body = (
        "The flat was bright, comfortable and clean, the neighbourhood was quiet, "
        "and everything was exactly as described in the listing."
    )
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as handle:
        handle.write(REVIEW_HEADER)
        for n in range(count):
            handle.write(f'{n % 500},{n},2025-06-01,{n},Reviewer,"{body} ({n})"\n')
    return path


class TestStreaming:
    def test_the_review_file_is_streamed_rather_than_read_into_memory(self, tmp_path: Path) -> None:
        """Peak memory must not track the size of the file.

        Fifty thousand rows is roughly 7 MB of text. A loader that read the file,
        or collected the parsed rows, would show that in its peak; streaming
        shows a few tens of kilobytes of row at a time.
        """
        path = write_reviews(tmp_path / "reviews.csv.gz", count=50_000)
        assert path.stat().st_size > 100_000

        stats = _RowStats()
        tracemalloc.start()
        try:
            baseline = tracemalloc.get_traced_memory()[0]
            newest = max(review.date for review in _parsed_reviews(path, stats))
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

        assert stats.seen == 50_000
        assert newest == date(2025, 6, 1)
        assert peak - baseline < 1_000_000, "the review file is being accumulated, not streamed"

    def test_a_row_is_available_before_the_file_has_been_read(self, tmp_path: Path) -> None:
        """A generator, not a list: the COPY starts while the file is still open."""
        path = write_reviews(tmp_path / "reviews.csv.gz", count=10_000)
        rows = _parsed_reviews(path, _RowStats())
        first = next(rows)
        assert first.comments.endswith("(0)")
        rows.close()


class TestFilesThatCannotBeLoaded:
    def test_a_missing_column_is_named_rather_than_rejecting_every_row(
        self, tmp_path: Path
    ) -> None:
        """Inside Airbnb renames columns between snapshots; the load says which."""
        path = tmp_path / "reviews.csv"
        path.write_text("listing_id,date,body\n1,2025-06-01,Lovely\n", encoding="utf-8")

        with path.open(encoding="utf-8", newline="") as handle, pytest.raises(LoadError) as caught:
            list(_dict_rows(handle, path, REQUIRED_REVIEW_COLUMNS))
        assert "comments" in str(caught.value)

    def test_an_empty_file_is_a_load_error_not_an_empty_load(self, tmp_path: Path) -> None:
        path = tmp_path / "reviews.csv"
        path.write_text("", encoding="utf-8")

        with path.open(encoding="utf-8", newline="") as handle, pytest.raises(LoadError):
            list(_dict_rows(handle, path, REQUIRED_REVIEW_COLUMNS))


class TestRejectedRowsAreCounted:
    def test_an_unusable_row_is_counted_by_column_and_the_rest_still_load(
        self, tmp_path: Path
    ) -> None:
        """A bad row is data. Rejections are reported, never silently dropped."""
        path = tmp_path / "reviews.csv"
        path.write_text(
            "listing_id,date,comments\n"
            "1,2025-06-01,Lovely\n"
            "2,not-a-date,Also lovely\n"
            "3,2025-06-03,\n",
            encoding="utf-8",
        )

        stats = _RowStats()
        kept = list(_parsed_reviews(path, stats))

        assert [review.source_listing_id for review in kept] == [1]
        assert stats.seen == 3
        assert stats.rejected == 2
        assert dict(stats.by_field) == {"date": 1, "comments": 1}
        assert stats.top_reasons(limit=1)[0][1] == 1

    def test_listings_are_counted_the_same_way(self, tmp_path: Path) -> None:
        path = tmp_path / "listings.csv"
        header = (
            "id,name,description,neighbourhood_cleansed,room_type,property_type,latitude,"
            "longitude,price,accommodates,bedrooms,beds,bathrooms,bathrooms_text,"
            "minimum_nights,number_of_reviews,review_scores_rating,review_scores_location,"
            "review_scores_cleanliness,instant_bookable,host_is_superhost,amenities\n"
        )
        good = (
            "1,Canal studio,Quiet,Hackney,Entire home/apt,Entire rental unit,51.54,-0.05,"
            '$150.00,2,1,1,1.0,1 bath,2,12,4.8,4.7,4.9,,t,"[""Wifi""]"\n'
        )
        # No minimum_nights: five London listings are like this, and the column
        # is NOT NULL because a listing without one cannot be booked or cited.
        missing_minimum_nights = (
            "2,Park room,Bright,Islington,Private room,Private room in home,51.55,-0.10,"
            '$90.00,1,1,1,1.0,1 shared bath,,4,4.5,4.4,4.6,,f,"[""Wifi""]"\n'
        )
        path.write_text(header + good + missing_minimum_nights, encoding="utf-8")

        stats = _RowStats()
        kept = list(_parsed_listings(path, stats))

        assert [listing.source_listing_id for listing in kept] == [1]
        assert stats.rejected == 1
        assert dict(stats.by_field) == {"minimum_nights": 1}


class TestSnapshotDate:
    def test_an_iso_date_is_accepted(self) -> None:
        assert _require_snapshot_date("2026-06-19") == date(2026, 6, 19)

    @pytest.mark.parametrize("raw", [None, "", "june 2026", "19-06-2026"])
    def test_a_load_refuses_to_run_without_one(self, raw: str | None) -> None:
        """Listing ids are not stable between releases, so undated data is unusable."""
        with pytest.raises(LoadError):
            _require_snapshot_date(raw)
