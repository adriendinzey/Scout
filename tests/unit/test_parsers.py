"""The messy cases the real snapshot actually contains.

Every value asserted here was taken from the London file rather than imagined:
the bathroom wordings are its 50 distinct spellings, the price shapes are the
seven it prints, and `instant_bookable` is empty in all 92,638 of its rows.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from typing import ClassVar

import pytest

from scout.data.parsers import (
    DROPPED_PERSONAL_COLUMNS,
    FLOAT4_MAX,
    INT2,
    OTHER_PROPERTY_TYPE,
    REQUIRED_LISTING_COLUMNS,
    REQUIRED_REVIEW_COLUMNS,
    ListingRow,
    RowParseError,
    canonical_spellings,
    collapse_property_type,
    kept_property_types,
    listing_row_field_names,
    missing_columns,
    normalize_amenity,
    parse_amenities,
    parse_bathrooms,
    parse_bool,
    parse_date,
    parse_float,
    parse_int,
    parse_listing_row,
    parse_price,
    parse_review_row,
    review_row_field_names,
)

# One row in the shape the snapshot writes, personal columns included, so the
# parsers can be shown dropping them rather than trusted to.
RAW_LISTING: dict[str, str | None] = {
    "id": "11551",
    "listing_url": "https://www.airbnb.com/rooms/11551",
    "name": "Artistic London Pied-a-Terre",
    "description": "Live like a local in this quintessential London flat.",
    "host_id": "43039",
    "host_url": "https://www.airbnb.com/users/show/43039",
    "host_name": "Adriano",
    "host_about": "I have lived in Brixton for twenty years.",
    "host_thumbnail_url": "https://a0.example/thumb.jpg",
    "host_picture_url": "https://a0.example/pic.jpg",
    "host_profile_id": "43039",
    "host_profile_url": "https://www.airbnb.com/users/show/43039",
    "host_is_superhost": "t",
    "neighbourhood_cleansed": "Lambeth",
    "latitude": "51.46095",
    "longitude": "-0.11758",
    "property_type": "Entire rental unit",
    "room_type": "Entire home/apt",
    "accommodates": "5",
    "bathrooms": "1.0",
    "bathrooms_text": "1 bath",
    "bedrooms": "1",
    "beds": "3",
    "amenities": '["Wifi", "Carbon monoxide alarm", "Coffee maker"]',
    "price": "$234.50",
    "minimum_nights": "1",
    "number_of_reviews": "196",
    "review_scores_rating": "4.55",
    "review_scores_location": "4.54",
    "review_scores_cleanliness": "4.57",
    "instant_bookable": "",
}

RAW_REVIEW: dict[str, str | None] = {
    "listing_id": "11551",
    "id": "30672",
    "date": "2010-03-21",
    "reviewer_id": "93896",
    "reviewer_name": "Shar-Lyn",
    "comments": "The flat was bright, comfortable and clean.",
}


class TestParsePrice:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$234.50", 234.50),
            ("$1,234.00", 1234.00),
            ("$99,999.00", 99999.00),
            ("$9.00", 9.00),
            ("234.50", 234.50),
            ("£1,234.00", 1234.00),
            ("$0.00", 0.0),
            ("  $75.00  ", 75.0),
        ],
    )
    def test_the_shapes_the_snapshot_prints(self, raw: str, expected: float) -> None:
        assert parse_price(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_no_price_is_none_rather_than_zero(self, raw: str | None) -> None:
        """A third of London's listings have no price, and none of them are free."""
        assert parse_price(raw) is None

    @pytest.mark.parametrize("raw", ["free", "$", "1.2.3"])
    def test_a_price_that_is_not_a_number_is_rejected(self, raw: str) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_price(raw)
        assert caught.value.field == "price"

    def test_a_negative_price_is_rejected(self) -> None:
        """The column has a CHECK; catching it here names the row instead."""
        with pytest.raises(RowParseError):
            parse_price("$-5.00")


class TestParseBathrooms:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1 bath", 1.0),
            ("2 baths", 2.0),
            ("1.5 baths", 1.5),
            ("1 shared bath", 1.0),
            ("2 shared baths", 2.0),
            ("1 private bath", 1.0),
            ("2.5 shared baths", 2.5),
            ("0 baths", 0.0),
            ("0 shared baths", 0.0),
            ("24 baths", 24.0),
            ("Half-bath", 0.5),
            ("Shared half-bath", 0.5),
            ("Private half-bath", 0.5),
            ("HALF-BATH", 0.5),
            ("1  bath", 1.0),
        ],
    )
    def test_every_wording_the_snapshot_uses(self, raw: str, expected: float) -> None:
        assert parse_bathrooms(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", "   ", None])
    def test_missing_is_none_so_the_numeric_column_can_fill_it(self, raw: str | None) -> None:
        assert parse_bathrooms(raw) is None

    @pytest.mark.parametrize("raw", ["one bath", "bath", "1 bathroom", "1.5"])
    def test_an_unknown_wording_is_rejected_rather_than_guessed(self, raw: str) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_bathrooms(raw)
        assert caught.value.field == "bathrooms_text"


class TestParseBool:
    @pytest.mark.parametrize(("raw", "expected"), [("t", True), ("f", False), ("T", True)])
    def test_the_sources_t_and_f(self, raw: str, expected: bool) -> None:
        assert parse_bool(raw, field="host_is_superhost") is expected

    @pytest.mark.parametrize("raw", ["", "  ", None])
    def test_empty_is_none_not_false(self, raw: str | None) -> None:
        """`instant_bookable` is empty for every row of the London snapshot.

        False would say every listing refuses instant booking, which is a claim
        the snapshot does not make.
        """
        assert parse_bool(raw, field="instant_bookable") is None

    @pytest.mark.parametrize("raw", ["yes", "1", "maybe"])
    def test_anything_else_is_rejected(self, raw: str) -> None:
        with pytest.raises(RowParseError):
            parse_bool(raw, field="instant_bookable")


class TestParseNumbers:
    @pytest.mark.parametrize(("raw", "expected"), [("5", 5), ("0", 0), (" 12 ", 12), ("-3", -3)])
    def test_integers(self, raw: str, expected: int) -> None:
        assert parse_int(raw, field="beds") == expected

    @pytest.mark.parametrize("raw", ["", None])
    def test_an_empty_integer_is_none(self, raw: str | None) -> None:
        assert parse_int(raw, field="bedrooms") is None

    @pytest.mark.parametrize("raw", ["1.5", "two", "1e3"])
    def test_a_non_integer_is_rejected(self, raw: str) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_int(raw, field="bedrooms")
        assert caught.value.field == "bedrooms"

    @pytest.mark.parametrize(("raw", "expected"), [("4.55", 4.55), ("5", 5.0), (" 0.0 ", 0.0)])
    def test_floats(self, raw: str, expected: float) -> None:
        assert parse_float(raw, field="review_scores_rating") == pytest.approx(expected)

    @pytest.mark.parametrize("raw", ["", None])
    def test_an_unrated_listing_is_none(self, raw: str | None) -> None:
        """A quarter of listings have no rating, and none of them scored zero."""
        assert parse_float(raw, field="review_scores_rating") is None


class TestValuesNoColumnCanHold:
    """Rejected as one bad row, because they would otherwise abort the COPY.

    None of these occur in the London snapshot. A damaged download or another
    city is a different matter, and "one row killed the load of ninety thousand"
    is an unhelpful way to find that out.
    """

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("bedrooms", "40000"),
            ("beds", "-40000"),
            ("accommodates", "40000"),
            ("minimum_nights", "99999999999"),
            ("number_of_reviews", "99999999999"),
            ("id", "99999999999999999999999"),
        ],
    )
    def test_an_integer_too_wide_for_its_column_is_rejected(self, column: str, value: str) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_listing_row({**RAW_LISTING, column: value})
        assert caught.value.field == column

    def test_a_value_inside_the_column_width_is_fine(self) -> None:
        assert parse_int("32767", field="bedrooms", fits=INT2) == 32767

    @pytest.mark.parametrize("value", ["nan", "inf", "-inf", "Infinity"])
    def test_a_non_finite_number_is_rejected(self, value: str) -> None:
        """NaN compares false against every filter: the listing would vanish."""
        with pytest.raises(RowParseError):
            parse_float(value, field="review_scores_rating")

    def test_a_price_too_large_for_the_column_is_rejected(self) -> None:
        with pytest.raises(RowParseError):
            parse_price("$1" + "0" * 40)

    def test_a_price_at_the_top_of_the_column_is_kept(self) -> None:
        assert parse_price(f"${FLOAT4_MAX:.0f}") == pytest.approx(FLOAT4_MAX)

    @pytest.mark.parametrize("column", ["name", "description", "amenities"])
    def test_a_nul_byte_in_text_is_rejected_rather_than_stripped(self, column: str) -> None:
        """Editing a listing's own words on the way past would be worse."""
        value = '["Wi\x00fi"]' if column == "amenities" else "Canal\x00side studio"
        with pytest.raises(RowParseError) as caught:
            parse_listing_row({**RAW_LISTING, column: value})
        assert caught.value.field == column

    def test_a_nul_byte_in_a_review_is_rejected(self) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_review_row({**RAW_REVIEW, "comments": "Lovely\x00 place"})
        assert caught.value.field == "comments"


class TestParseDate:
    def test_an_iso_date(self) -> None:
        assert parse_date("2010-03-21") == date(2010, 3, 21)

    @pytest.mark.parametrize("raw", ["", None, "21/03/2010", "March 2010"])
    def test_anything_else_is_rejected(self, raw: str | None) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_date(raw)
        assert caught.value.field == "date"


class TestParseAmenities:
    def test_a_json_list_becomes_names(self) -> None:
        assert parse_amenities('["Wifi", "Kitchen"]') == ("Wifi", "Kitchen")

    @pytest.mark.parametrize("raw", ["", "  ", None])
    def test_no_amenities_is_an_empty_tuple(self, raw: str | None) -> None:
        assert parse_amenities(raw) == ()

    def test_source_order_is_kept(self) -> None:
        """The order is what the listing wrote; the document template reads it."""
        assert parse_amenities('["Kitchen", "Wifi", "Iron"]') == ("Kitchen", "Wifi", "Iron")

    def test_one_amenity_written_twice_is_one_amenity(self) -> None:
        assert parse_amenities('["Wifi", "WIFI", "wifi"]') == ("Wifi",)

    def test_spacing_and_unicode_are_normalized(self) -> None:
        curly = "\u2019"
        assert parse_amenities(f'["Pack  {curly}n play", "Pack {curly}n play"]') == (
            f"Pack {curly}n play",
        )

    def test_blank_entries_are_dropped(self) -> None:
        assert parse_amenities('["Wifi", "", "   "]') == ("Wifi",)

    @pytest.mark.parametrize("raw", ["not json", "{", '{"Wifi": true}', '["Wifi", 3]'])
    def test_a_list_that_is_not_a_list_of_strings_is_rejected(self, raw: str) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_amenities(raw)
        assert caught.value.field == "amenities"

    def test_normalize_leaves_the_wording_alone(self) -> None:
        """Only spelling is settled here; which spelling wins is decided on counts."""
        assert normalize_amenity("  Dedicated   workspace ") == "Dedicated workspace"
        assert normalize_amenity("SONOS sound system") == "SONOS sound system"


class TestCanonicalSpellings:
    def test_the_most_common_spelling_wins(self) -> None:
        counts = {"Sonos sound system": 40, "SONOS sound system": 3, "SOnos sound system": 1}
        assert canonical_spellings(counts) == {"sonos sound system": "Sonos sound system"}

    def test_a_tie_breaks_alphabetically_rather_than_on_read_order(self) -> None:
        """Two loads of the same file must produce the same lookup table."""
        counts = {"Wifi": 5, "WiFi": 5}
        assert canonical_spellings(counts) == {"wifi": "WiFi"}
        assert canonical_spellings(dict(reversed(list(counts.items())))) == {"wifi": "WiFi"}

    def test_distinct_amenities_keep_their_own_entries(self) -> None:
        assert canonical_spellings({"Wifi": 2, "Kitchen": 1}) == {
            "wifi": "Wifi",
            "kitchen": "Kitchen",
        }

    def test_no_amenities_is_no_entries(self) -> None:
        assert canonical_spellings({}) == {}


class TestPropertyTypes:
    COUNTS: ClassVar[dict[str, int]] = {
        "Entire rental unit": 40129,
        "Boat": 79,
        "Lighthouse": 1,
        "Cave": 1,
    }

    def test_types_below_the_threshold_are_dropped_from_the_lookup(self) -> None:
        assert kept_property_types(self.COUNTS, min_listings=50) == frozenset(
            {"Entire rental unit", "Boat"}
        )

    def test_a_threshold_of_one_keeps_everything(self) -> None:
        assert kept_property_types(self.COUNTS, min_listings=1) == frozenset(self.COUNTS)

    def test_a_kept_type_is_left_alone(self) -> None:
        kept = kept_property_types(self.COUNTS, min_listings=50)
        assert collapse_property_type("Boat", kept) == "Boat"

    def test_a_rare_type_becomes_other(self) -> None:
        kept = kept_property_types(self.COUNTS, min_listings=50)
        assert collapse_property_type("Lighthouse", kept) == OTHER_PROPERTY_TYPE


class TestMissingColumns:
    def test_a_header_with_everything_is_missing_nothing(self) -> None:
        assert missing_columns(list(REQUIRED_REVIEW_COLUMNS), REQUIRED_REVIEW_COLUMNS) == []

    def test_a_renamed_column_is_named(self) -> None:
        """Inside Airbnb renames columns between snapshots; the load says which."""
        header = ["listing_id", "date", "body"]
        assert missing_columns(header, REQUIRED_REVIEW_COLUMNS) == ["comments"]

    def test_an_empty_file_is_missing_all_of_them(self) -> None:
        assert missing_columns(None, REQUIRED_REVIEW_COLUMNS) == sorted(REQUIRED_REVIEW_COLUMNS)


class TestParseListingRow:
    def test_a_real_row_becomes_a_listing(self) -> None:
        listing = parse_listing_row(RAW_LISTING)

        assert listing.source_listing_id == 11551
        assert listing.name == "Artistic London Pied-a-Terre"
        assert listing.neighbourhood == "Lambeth"
        assert listing.room_type == "Entire home/apt"
        assert listing.property_type == "Entire rental unit"
        assert listing.price_gbp == pytest.approx(234.50)
        assert listing.accommodates == 5
        assert listing.bathrooms == pytest.approx(1.0)
        assert listing.rating == pytest.approx(4.55)
        assert listing.host_is_superhost is True
        assert listing.amenities == ("Wifi", "Carbon monoxide alarm", "Coffee maker")

    def test_an_empty_instant_bookable_is_unknown(self) -> None:
        assert parse_listing_row(RAW_LISTING).instant_bookable is None

    def test_a_missing_description_is_none(self) -> None:
        listing = parse_listing_row({**RAW_LISTING, "description": ""})
        assert listing.description is None

    def test_the_numeric_bathrooms_column_fills_in_for_missing_text(self) -> None:
        """`bathrooms_text` is empty for 134 London listings; some have the number."""
        listing = parse_listing_row({**RAW_LISTING, "bathrooms_text": "", "bathrooms": "2.5"})
        assert listing.bathrooms == pytest.approx(2.5)

    def test_neither_bathrooms_column_is_none(self) -> None:
        listing = parse_listing_row({**RAW_LISTING, "bathrooms_text": "", "bathrooms": ""})
        assert listing.bathrooms is None

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("id", ""),
            ("name", ""),
            ("neighbourhood_cleansed", ""),
            ("room_type", ""),
            ("property_type", ""),
            ("latitude", ""),
            ("longitude", ""),
            ("accommodates", ""),
            ("minimum_nights", ""),
            ("number_of_reviews", ""),
        ],
    )
    def test_a_listing_missing_something_it_is_searched_by_is_rejected(
        self, column: str, value: str
    ) -> None:
        """Five London listings have no minimum_nights. Half a listing is not stored."""
        with pytest.raises(RowParseError) as caught:
            parse_listing_row({**RAW_LISTING, column: value})
        assert caught.value.field == column

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("latitude", "91.0"),
            ("longitude", "-181.0"),
            ("accommodates", "0"),
            ("minimum_nights", "0"),
            ("number_of_reviews", "-1"),
        ],
    )
    def test_a_value_the_column_constraints_would_refuse_is_rejected_here(
        self, column: str, value: str
    ) -> None:
        """Named as one row, rather than as a CHECK violation aborting the COPY."""
        with pytest.raises(RowParseError):
            parse_listing_row({**RAW_LISTING, column: value})

    def test_a_rejection_says_which_column_was_wrong(self) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_listing_row({**RAW_LISTING, "price": "cheap"})
        assert caught.value.field == "price"
        assert "cheap" in str(caught.value)


class TestParseReviewRow:
    def test_a_real_row_becomes_a_citation(self) -> None:
        review = parse_review_row(RAW_REVIEW)
        assert review.source_listing_id == 11551
        assert review.date == date(2010, 3, 21)
        assert review.comments == "The flat was bright, comfortable and clean."

    @pytest.mark.parametrize(
        ("column", "value"), [("comments", ""), ("date", ""), ("listing_id", "")]
    )
    def test_a_review_that_cannot_be_cited_is_rejected(self, column: str, value: str) -> None:
        with pytest.raises(RowParseError) as caught:
            parse_review_row({**RAW_REVIEW, column: value})
        assert caught.value.field == column


class TestPersonalFieldsAreDroppedAtParseTime:
    """docs/DATA.md section 4, asserted rather than trusted.

    The check is on the parsed row, not on the database: by the time a column
    could be caught there, the value has already been written. These rows are
    the only thing the loader can write, and they have nowhere to put a name.
    """

    def test_no_listing_field_is_a_personal_one(self) -> None:
        assert listing_row_field_names() & DROPPED_PERSONAL_COLUMNS == frozenset()

    def test_no_review_field_is_a_personal_one(self) -> None:
        assert review_row_field_names() & DROPPED_PERSONAL_COLUMNS == frozenset()

    def test_the_host_name_in_a_row_reaches_nothing(self) -> None:
        listing = parse_listing_row(RAW_LISTING)
        stored = repr(asdict(listing))
        for value in ("Adriano", "43039", "airbnb.com", "Brixton for twenty years"):
            assert value not in stored

    def test_the_reviewer_name_in_a_row_reaches_nothing(self) -> None:
        review = parse_review_row(RAW_REVIEW)
        stored = repr(asdict(review))
        assert "Shar-Lyn" not in stored
        assert "93896" not in stored

    def test_the_upstream_review_id_is_not_kept_either(self) -> None:
        """It resolves back to a reviewer's profile, like host_url does."""
        assert "id" not in review_row_field_names()
        assert parse_review_row({**RAW_REVIEW, "id": "30672"}).comments == RAW_REVIEW["comments"]

    def test_host_is_superhost_is_kept_because_it_describes_the_listing(self) -> None:
        assert "host_is_superhost" in listing_row_field_names()
        assert "host_is_superhost" not in DROPPED_PERSONAL_COLUMNS


def test_a_listing_row_names_exactly_the_columns_the_loader_writes() -> None:
    """A field added here without a column to hold it fails at COPY, not in review."""
    assert listing_row_field_names() == {
        "source_listing_id",
        "name",
        "description",
        "neighbourhood",
        "room_type",
        "property_type",
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
    }


def test_the_sample_row_carries_every_column_the_parser_requires() -> None:
    """Otherwise the tests above would be parsing a file shape that cannot exist."""
    assert missing_columns(list(RAW_LISTING), REQUIRED_LISTING_COLUMNS) == []


def test_a_listing_row_is_immutable() -> None:
    """Nothing between parsing and COPY may edit a row into a different listing."""
    listing = parse_listing_row(RAW_LISTING)
    with pytest.raises((AttributeError, TypeError)):
        listing.name = "Something else"  # type: ignore[misc]
    assert isinstance(listing, ListingRow)
