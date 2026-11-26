"""The words the Parse node sends, kept apart from the code that sends them.

Everything here except :func:`retry_feedback` lands in the system prompt, which
is the cached prefix of every parse. A change to this file is a change to every
request, so it is written to be read as a diff. Two rules keep the prefix
cacheable:

* **Nothing here varies per query.** The user's request travels in the message
  list, after the cached prefix, never inside it.
* **The vocabulary is rendered from sorted names.** A set rendered in whatever
  order the database or a hash table produced would differ between processes,
  and the cache would miss on every run with no symptom but the bill.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

INSTRUCTIONS: Final = """\
You turn one request for a short-term rental into a structured search for Scout. \
Reply with a single JSON object that follows the response schema.

Scout searches two ways at once. Filters are exact database conditions. The \
semantic query is ranked by meaning against each listing's description, amenities \
and reviews. Decide which parts of the request belong to which, and report \
anything Scout cannot search for at all.

Filters
- Set a filter only for something the user asked for. Leave every other filter \
out. Never invent a bound the user did not give.
- Unconstrained means leaving a filter out, never setting it to zero. \
min_bedrooms 0 is not "any number of bedrooms": it still excludes every listing \
whose bedroom count is unknown.
- A listing with no value for a field satisfies no condition on it. A rating \
floor excludes every unrated listing, and a price bound excludes every listing \
with no published price. That is why a filter is only for an explicit ask.
- Prices are per night, in GBP. A total for a stated number of nights is divided \
by the nights. A price in another currency keeps its number unconverted, and is \
also recorded as an unsupported constraint saying so.
- Ratings and location scores are on a 0 to 5 scale. One vague phrase has a fixed \
meaning: "well reviewed" or "highly rated" is min_rating 4.5. No other vague word \
sets a bound.
- The number of guests is min_accommodates. The length of the stay is \
max_minimum_nights, because a listing that insists on more nights than the stay \
cannot be booked for it.
- Several neighbourhoods, room types or property types mean any of them. Several \
amenities mean all of them.
- Names are copied exactly from the vocabulary below, and used only when the name \
means what the user asked for.

Not filters
- Character and setting go in semantic_query only: quiet, cosy, stylish, near the \
water, good for working remotely. There is no column for them.
- Scout has no dates or availability, and no map. It cannot place a landmark, a \
station, a street, or any area that is not in the neighbourhood list. Each of \
those is an unsupported constraint, and its words stay in semantic_query too, \
where they still help ranking.
- An amenity that is not in the amenity vocabulary is an unsupported constraint, \
and belongs in semantic_query as well.
- A contradictory request is parsed as written. "A cheap luxury penthouse under \
£40" is max_price 40, with the rest left to the semantic query. Do not resolve \
the contradiction; the search reports what it finds.

semantic_query
- A short phrase describing the place wanted, in the user's terms, without the \
numbers the filters already carry. Never empty: if the request is all filters, \
describe the place in a few words.

unsupported_constraints
- Everything the user asked for that no filter expresses exactly, apart from the \
descriptive qualities semantic_query captures. Nothing the user asked for is \
dropped.
- constraint is the user's own words. reason says in a few words why Scout \
cannot filter on it.
"""

# Until amenities are chosen for indexing there are none to list. Saying so
# outright is what stops the model reaching for an amenity filter anyway.
_NO_AMENITIES: Final = (
    "- (none: this database has no amenity filters, so every amenity the user asks "
    "for is an unsupported constraint)"
)


def _section(heading: str, names: Iterable[str], *, when_empty: str) -> str:
    """One vocabulary list, headed by the response field that draws from it."""
    ordered = sorted(names)
    items = [f"- {name}" for name in ordered] or [when_empty]
    return "\n".join([f"{heading} ({len(ordered)}):", *items])


def render_vocabulary(
    *,
    city: str,
    neighbourhoods: Iterable[str],
    room_types: Iterable[str],
    property_types: Iterable[str],
    amenities: Iterable[str],
) -> str:
    """The grounding block: every name a filter may use, sorted.

    Sorted by code point rather than by the database's collation, so the bytes
    depend only on the names and never on the server's locale settings.
    """
    sections = [
        f"Vocabulary\n\nEvery listing is in {city}. Names are spelled exactly as "
        f"Scout stores them, and a filter may use no other.",
        _section("neighbourhoods", neighbourhoods, when_empty="- (none)"),
        _section("room_types", room_types, when_empty="- (none)"),
        _section("property_types", property_types, when_empty="- (none)"),
        _section("amenities", amenities, when_empty=_NO_AMENITIES),
    ]
    return "\n\n".join(sections)


def system_prompt(vocabulary: str) -> str:
    """The full system prompt: the fixed instructions, then the grounding block."""
    return f"{INSTRUCTIONS}\n{vocabulary}\n"


def retry_feedback(problem: str) -> str:
    """The follow-up turn after a response that failed validation.

    Says that nothing was searched, so the model does not read the rejection as
    an empty result and start loosening filters to fix it.
    """
    return (
        "That response was rejected before anything was searched. "
        f"The problem: {problem}\n"
        "Reply again with one JSON object that follows the schema and fixes it."
    )
