"""The filter vocabulary: what Scout can express as a database predicate.

One Pydantic model is the contract three separate things share -- what the
parser is allowed to emit, what a tool validates its arguments against, and what
the query builder knows how to translate. Keeping it in one place is what turns
an invented column into a validation error instead of a silent no-op.

Two properties are deliberate.

**`None` means unconstrained.** It is never shorthand for a permissive default:
an absent price ceiling emits no price condition at all rather than
``price_gbp <= inf``, so "the user said nothing about price" stays
distinguishable from "the user will pay anything".

**A filter set is frozen and canonically ordered.** Set-valued fields are sorted
and deduplicated on validation, so two requests for the same thing compare and
hash equal however the model happened to order them. That is what lets the agent
loop recognise a repeated call instead of paying for the same search twice, and
it keeps the fan-out branch order stable across runs.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Lookup-table ids are generated identities, so they start at 1; a zero or
# negative id is a parse error rather than a query that returns nothing.
LookupId = Annotated[int, Field(ge=1)]

# Ratings and sub-scores share the source's 0-5 scale.
MAX_RATING = 5.0

# A radius larger than this is not a neighbourhood search any more, and the
# bounding box it implies would cover most of a country. Scout searches one
# city, so anything beyond city scale is a mistake worth rejecting where the
# model can still correct it.
MAX_RADIUS_KM = 200.0


class NearFilter(BaseModel):
    """A point and a radius in kilometres, used as a coarse area constraint.

    Coordinates in the corpus are deliberately offset by up to ~150 m upstream so
    a listing cannot be pinpointed, which puts a floor under how precise any
    distance filter can honestly be. This is an area, never an address.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    radius_km: float = Field(gt=0.0, le=MAX_RADIUS_KM)


class Filters(BaseModel):
    """A validated, structured version of everything a request constrains.

    Every field defaults to unconstrained. Set-valued fields (`neighbourhood_ids`,
    `room_type_ids`, `property_type_ids`) are **disjunctive** -- any of these --
    and become one query per value, because the index cannot push down `OR`.
    `amenities` is **conjunctive**: a listing must have all of them.

    Unknown fields are rejected rather than ignored, so a hallucinated column
    fails loudly at the boundary instead of quietly narrowing nothing.

    A zero bound is not the same as no bound. `min_bedrooms=0` constrains
    nothing arithmetically but still emits `bedrooms >= 0`, which excludes every
    listing whose bedroom count is unknown; `min_bedrooms=None` excludes
    nothing. A built plan reports the difference in `null_excluding_columns`,
    and a caller loosening a filter should clear it rather than zero it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # --- disjunctive: any of these ------------------------------------------
    neighbourhood_ids: tuple[LookupId, ...] = ()
    room_type_ids: tuple[LookupId, ...] = ()
    property_type_ids: tuple[LookupId, ...] = ()

    # --- price ---------------------------------------------------------------
    # Nightly price in USD, as loaded. A listing with no price satisfies neither
    # bound; see `null_excluding_columns` on a built plan.
    min_price: float | None = Field(default=None, ge=0.0)
    max_price: float | None = Field(default=None, ge=0.0)

    # --- capacity ------------------------------------------------------------
    min_accommodates: int | None = Field(default=None, ge=1)
    min_bedrooms: int | None = Field(default=None, ge=0)
    min_beds: int | None = Field(default=None, ge=0)
    min_bathrooms: float | None = Field(default=None, ge=0.0)
    # A stay is impossible if the listing demands more nights than the guest
    # wants, so the guest's trip length is an upper bound on `minimum_nights`.
    max_minimum_nights: int | None = Field(default=None, ge=1)

    # --- quality -------------------------------------------------------------
    min_rating: float | None = Field(default=None, ge=0.0, le=MAX_RATING)
    min_location_score: float | None = Field(default=None, ge=0.0, le=MAX_RATING)
    min_reviews: int | None = Field(default=None, ge=0)

    # --- booking -------------------------------------------------------------
    instant_bookable: bool | None = None
    host_is_superhost: bool | None = None

    # --- conjunctive: all of these -------------------------------------------
    # Amenity names as the lookup table spells them. Which ones have an indexed
    # boolean column is a property of the schema, passed in at build time rather
    # than known here.
    amenities: tuple[str, ...] = ()

    # --- area ----------------------------------------------------------------
    near: NearFilter | None = None

    @field_validator("neighbourhood_ids", "room_type_ids", "property_type_ids")
    @classmethod
    def _canonical_ids(cls, ids: tuple[int, ...]) -> tuple[int, ...]:
        """Sort and deduplicate, so equal filter sets are equal objects."""
        return tuple(sorted(set(ids)))

    @field_validator("amenities")
    @classmethod
    def _canonical_amenities(cls, amenities: tuple[str, ...]) -> tuple[str, ...]:
        """Sort and deduplicate; a blank name is a mistake, not an empty filter."""
        cleaned = [name.strip() for name in amenities]
        if any(not name for name in cleaned):
            raise ValueError("amenity names must not be blank")
        return tuple(sorted(set(cleaned)))

    @model_validator(mode="after")
    def _price_range_is_satisfiable(self) -> Filters:
        """Reject an empty price window where the caller can still fix it.

        `min_price > max_price` matches nothing by construction. Left to run it
        would look exactly like an over-constrained search, and the loop would
        spend its budget relaxing other filters to fix a contradiction.
        """
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                f"min_price ({self.min_price}) exceeds max_price ({self.max_price}); "
                f"no listing can satisfy both"
            )
        return self

    def is_empty(self) -> bool:
        """True when nothing is constrained -- a search across the whole corpus."""
        return self == Filters()
