"""The circle-to-box conversion, checked against an independent distance formula.

The box is built with flat-earth scaling; these tests measure it with haversine,
so a mistake in the scaling cannot hide behind the same arithmetic that produced
it. The property that matters is that the box is a **superset** of the circle:
being slightly too wide costs a few extra candidates, being too narrow silently
drops listings that genuinely qualify.
"""

from __future__ import annotations

import math

import pytest

from scout.retrieval.filters import MAX_RADIUS_KM, NearFilter
from scout.retrieval.query import KM_PER_DEGREE_LATITUDE, bounding_box

LONDON_LATITUDE = 51.5
LONDON_LONGITUDE = -0.12

EARTH_RADIUS_KM = 6371.0


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance between two (latitude, longitude) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    h = (
        math.sin((lat2 - lat1) / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def test_latitude_delta_is_radius_over_111_km() -> None:
    box = bounding_box(NearFilter(latitude=0.0, longitude=0.0, radius_km=11.1))
    south, north = box.latitude
    assert north - 0.0 == pytest.approx(11.1 / KM_PER_DEGREE_LATITUDE)
    assert 0.0 - south == pytest.approx(11.1 / KM_PER_DEGREE_LATITUDE)


def test_longitude_is_not_compressed_at_the_equator() -> None:
    """cos(0) is 1, so the box is square only here -- the case that hides bugs."""
    box = bounding_box(NearFilter(latitude=0.0, longitude=0.0, radius_km=10.0))
    assert box.longitude is not None
    latitude_span = box.latitude[1] - box.latitude[0]
    longitude_span = box.longitude[1] - box.longitude[0]
    # Square to within a millionth: the circle still reaches a hair off the
    # equator, where a degree of longitude is fractionally shorter.
    assert longitude_span == pytest.approx(latitude_span, rel=1e-5)
    assert longitude_span >= latitude_span


def test_longitude_is_compressed_at_londons_latitude() -> None:
    """A degree of longitude is ~69 km at 51.5 deg N, not ~111 km.

    Using the latitude scaling for both would make the box ~1.6x too narrow
    east-west, quietly excluding listings inside the requested radius.
    """
    radius_km = 2.0
    box = bounding_box(
        NearFilter(latitude=LONDON_LATITUDE, longitude=LONDON_LONGITUDE, radius_km=radius_km)
    )
    assert box.longitude is not None

    km_per_degree_longitude = KM_PER_DEGREE_LATITUDE * math.cos(math.radians(LONDON_LATITUDE))
    assert km_per_degree_longitude == pytest.approx(69.1, abs=0.1)

    # Scaled at the northern edge of the band rather than at 51.5 exactly, so
    # the box stays a superset; at city scale the two differ by ~0.04%.
    expected_delta = radius_km / (KM_PER_DEGREE_LATITUDE * math.cos(math.radians(box.latitude[1])))
    assert expected_delta == pytest.approx(radius_km / km_per_degree_longitude, rel=1e-3)
    assert box.longitude[1] - LONDON_LONGITUDE == pytest.approx(expected_delta)
    assert LONDON_LONGITUDE - box.longitude[0] == pytest.approx(expected_delta)

    latitude_span = box.latitude[1] - box.latitude[0]
    longitude_span = box.longitude[1] - box.longitude[0]
    assert longitude_span == pytest.approx(latitude_span * 1.607, rel=1e-3)


@pytest.mark.parametrize("latitude", [0.0, 25.0, 51.5, 60.0, -33.9, 88.0])
@pytest.mark.parametrize("radius_km", [0.5, 2.0, 10.0, 50.0])
def test_the_box_encloses_the_circle(latitude: float, radius_km: float) -> None:
    """Every cardinal edge sits at least `radius_km` from the centre."""
    centre = (latitude, 10.0)
    box = bounding_box(NearFilter(latitude=latitude, longitude=10.0, radius_km=radius_km))
    assert box.longitude is not None

    assert haversine_km(centre, (box.latitude[0], centre[1])) >= radius_km
    assert haversine_km(centre, (box.latitude[1], centre[1])) >= radius_km
    assert haversine_km(centre, (centre[0], box.longitude[0])) >= radius_km
    assert haversine_km(centre, (centre[0], box.longitude[1])) >= radius_km


@pytest.mark.parametrize("latitude", [89.0, -89.0, 89.9])
def test_a_circle_reaching_a_pole_covers_every_longitude(latitude: float) -> None:
    """Past a pole the circle wraps onto the far meridian.

    At 89 deg N a 200 km radius reaches (89.5, 180) -- 167 km away, comfortably
    inside -- while any bounded longitude range excludes it. Keeping a bound
    there would make the box a subset of the circle in one direction, which is
    the one failure the superset property exists to rule out.
    """
    box = bounding_box(NearFilter(latitude=latitude, longitude=0.0, radius_km=200.0))
    assert box.latitude[0] <= -90.0 or box.latitude[1] >= 90.0
    assert box.longitude is None


def test_the_far_side_of_a_pole_is_genuinely_inside_the_radius() -> None:
    """The premise of the test above, measured rather than asserted."""
    assert haversine_km((89.0, 0.0), (89.5, 180.0)) == pytest.approx(166.8, abs=0.5)


def test_a_vanishing_cosine_widens_the_box_rather_than_capping_it() -> None:
    """The guard must fail open, not narrow.

    Within metres of a pole the longitude delta diverges. Capping the divisor
    bounds the delta and makes the box a *subset* of the circle -- at 89.99999
    deg it emitted 0.90 deg where the circle needs 5.16 -- which is the one
    direction the superset property rules out.
    """
    box = bounding_box(NearFilter(latitude=89.99999, longitude=0.0, radius_km=0.0001))
    assert box.latitude[1] < 90.0, "the pole branch must not be what covers this"
    assert box.longitude is None


def test_a_circle_well_clear_of_a_pole_keeps_a_bounded_longitude() -> None:
    """The other side of the boundary: 50 km from 80 deg N stays bounded."""
    box = bounding_box(NearFilter(latitude=80.0, longitude=0.0, radius_km=50.0))
    assert box.latitude[1] < 90.0
    assert box.longitude == (pytest.approx(-2.7152, abs=1e-4), pytest.approx(2.7152, abs=1e-4))


def test_the_box_is_not_absurdly_wider_than_the_circle() -> None:
    """A superset, but a tight one -- a loose box wastes the candidate pool."""
    radius_km = 5.0
    centre = (LONDON_LATITUDE, LONDON_LONGITUDE)
    box = bounding_box(
        NearFilter(latitude=LONDON_LATITUDE, longitude=LONDON_LONGITUDE, radius_km=radius_km)
    )
    assert box.longitude is not None
    for edge in (
        (box.latitude[1], centre[1]),
        (centre[0], box.longitude[1]),
    ):
        assert haversine_km(centre, edge) < radius_km * 1.01


def test_latitude_is_clamped_to_the_poles() -> None:
    box = bounding_box(NearFilter(latitude=89.9, longitude=0.0, radius_km=100.0))
    assert box.latitude[1] == 90.0
    box = bounding_box(NearFilter(latitude=-89.9, longitude=0.0, radius_km=100.0))
    assert box.latitude[0] == -90.0


def test_a_pole_does_not_divide_by_zero() -> None:
    """cos(90 deg) is ~0; the guard turns a blow-up into "every longitude"."""
    box = bounding_box(NearFilter(latitude=90.0, longitude=0.0, radius_km=1.0))
    assert box.longitude is None


@pytest.mark.parametrize("longitude", [179.9, -179.9])
def test_a_box_that_would_wrap_the_antimeridian_covers_every_longitude(longitude: float) -> None:
    """Splitting the wrap into two ranges would need OR, which does not push down.

    Widening to every longitude keeps the predicate in the index and keeps the
    superset property; the alternative silently drops the half that wrapped.
    """
    box = bounding_box(NearFilter(latitude=0.0, longitude=longitude, radius_km=50.0))
    assert box.longitude is None


def test_the_widest_permitted_radius_is_still_a_bounded_box_at_city_latitudes() -> None:
    """Degenerating to "every longitude" is the exception, not the common case."""
    box = bounding_box(
        NearFilter(latitude=LONDON_LATITUDE, longitude=LONDON_LONGITUDE, radius_km=MAX_RADIUS_KM)
    )
    assert box.longitude is not None
    assert box.longitude[1] - box.longitude[0] < 10.0


def test_a_high_latitude_box_widens_but_stays_bounded_until_it_reaches_the_pole() -> None:
    """Longitude spans grow fast near a pole; None is the limit, not the rule."""
    assert bounding_box(NearFilter(latitude=85.0, longitude=0.0, radius_km=200.0)).longitude == (
        pytest.approx(-32.296, abs=1e-3),
        pytest.approx(32.296, abs=1e-3),
    )
    assert bounding_box(NearFilter(latitude=89.9, longitude=0.0, radius_km=200.0)).longitude is None
