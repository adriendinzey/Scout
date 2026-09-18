-- The tables Scout searches.
--
-- Column types here are not a style choice. Brindle pushes a predicate into the
-- vector index only when the column is bool, int2, int4, int8, float4 or
-- float8; text and timestamp columns are refused outright at CREATE INDEX. So
-- neighbourhood, room type and property type are integer foreign keys into
-- lookup tables rather than the text they arrived as, and every other
-- filterable attribute is a number or a boolean.
--
-- Index key budget. PostgreSQL allows at most 32 key columns per index. Scout's
-- index spends 1 on the embedding and 16 on the filterable columns created
-- here -- neighbourhood_id, room_type_id, property_type_id, price_usd,
-- accommodates, bedrooms, beds, bathrooms, minimum_nights, rating,
-- location_score, number_of_reviews, instant_bookable, host_is_superhost,
-- latitude, longitude -- for a total of 17. That leaves headroom for exactly
-- **15 amenity boolean columns**, which is the budget the amenity selection has
-- to fit inside. cleanliness_score is stored for display and deliberately kept
-- out of the index rather than spending another slot.
--
-- No index is created in this migration. Brindle stores an index as one blob
-- and rewrites the whole thing on every write, so an index that exists during
-- the bulk load makes the load pathologically slow. Rows land first; the index
-- is built afterwards, once.
--
-- Personal data is absent by construction, not nulled out afterwards: there is
-- no host identity column on listings and no reviewer column on reviews.
-- docs/DATA.md section 4 is the binding list of what gets dropped at parse time.

-- ----------------------------------------------------------------- lookups --
-- Text that would otherwise be unfilterable, exchanged for a small integer.

CREATE TABLE neighbourhoods (
    id   int4 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);

CREATE TABLE room_types (
    id   int2 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);

CREATE TABLE property_types (
    id   int2 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE
);

CREATE TABLE amenities (
    id   int4 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL UNIQUE,
    -- Set for the amenities that earn one of the 15 boolean columns; the rest
    -- stay searchable only through listings.amenities and the embedded text.
    is_indexed bool NOT NULL DEFAULT false
);

-- ---------------------------------------------------------------- listings --
-- One row per searchable unit.
--
-- Nullability is load-bearing, because a NULL satisfies no comparison: with
-- rating nullable, `rating >= 4.5` silently excludes every listing that has no
-- rating yet rather than including it. Parse and Check both depend on knowing
-- that, and the answer says so when a quality filter was applied.
--
-- Nullable on purpose, because the source genuinely omits them: price_usd,
-- bedrooms, beds, bathrooms, rating, location_score, cleanliness_score,
-- host_is_superhost and description. Nullable only until the pipeline fills
-- them: doc_text and embedding. Everything else is NOT NULL, and a row missing
-- one of those is a row that cannot be searched or cited, so the load rejects
-- it rather than storing a half listing.

CREATE TABLE listings (
    id                int4   GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    -- The upstream listing id. Kept so a reload can match rows and so a result
    -- can be traced back to the snapshot it came from; it identifies a
    -- property, not a person.
    source_listing_id int8   NOT NULL UNIQUE,

    name              text   NOT NULL,
    description       text,

    neighbourhood_id  int4   NOT NULL REFERENCES neighbourhoods (id),
    room_type_id      int2   NOT NULL REFERENCES room_types (id),
    property_type_id  int2   NOT NULL REFERENCES property_types (id),

    -- Already offset by up to ~150m upstream, deliberately, so a listing cannot
    -- be pinpointed. Coarse bounding-box filtering only; never an address.
    latitude          float8 NOT NULL,
    longitude         float8 NOT NULL,

    price_usd         float4,
    accommodates      int2   NOT NULL,
    bedrooms          int2,
    beds              int2,
    bathrooms         float4,
    minimum_nights    int4   NOT NULL,

    rating            float4,
    location_score    float4,
    cleanliness_score float4,
    number_of_reviews int4   NOT NULL,

    instant_bookable  bool   NOT NULL,
    -- A property of the listing's service level, not a personal detail, and a
    -- filter guests actually use.
    host_is_superhost bool,

    -- The full normalized amenity list. Not filterable through the index -- a
    -- text[] cannot be an index key column -- so it is for display and for
    -- building the embedded document.
    amenities         text[] NOT NULL DEFAULT '{}',

    -- The exact string that was embedded, stored so embeddings can be
    -- regenerated and debugged when the document template changes.
    doc_text          text,
    -- brindle_vector carries no dimension modifier, so width is enforced when
    -- the index is built rather than by the column type.
    embedding         brindle_vector,

    CONSTRAINT listings_latitude_in_range      CHECK (latitude BETWEEN -90 AND 90),
    CONSTRAINT listings_longitude_in_range     CHECK (longitude BETWEEN -180 AND 180),
    CONSTRAINT listings_price_not_negative     CHECK (price_usd >= 0),
    CONSTRAINT listings_accommodates_positive  CHECK (accommodates > 0),
    CONSTRAINT listings_minimum_nights_at_least_one CHECK (minimum_nights >= 1),
    CONSTRAINT listings_review_count_not_negative   CHECK (number_of_reviews >= 0)
);

COMMENT ON TABLE listings IS
    'One row per searchable listing. Carries no host identity: host_id, '
    'host_name, host_about, host_url, host_thumbnail_url, host_picture_url and '
    'listing_url are dropped while parsing, before the first INSERT.';

-- ----------------------------------------------------------------- reviews --
-- Stored to be cited, not to be searched: the unit of search is the listing.

CREATE TABLE reviews (
    -- Generated, not the upstream review id. An upstream id resolves back to a
    -- reviewer's profile on the public site, which is the same reason host_url
    -- and listing_url are dropped.
    id         int8 GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    listing_id int4 NOT NULL REFERENCES listings (id) ON DELETE CASCADE,
    date       date NOT NULL,
    comments   text NOT NULL
);

-- Reviews are always fetched for a listing Scout is about to cite.
CREATE INDEX reviews_listing_id_idx ON reviews (listing_id);

COMMENT ON TABLE reviews IS
    'Review text kept for citation. Carries no reviewer_id and no '
    'reviewer_name; both are dropped while parsing. Review bodies still pass '
    'through the name scrubber before they are embedded or displayed.';
