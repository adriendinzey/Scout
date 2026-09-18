-- What the snapshot actually contains, applied to the schema 0001 assumed.
--
-- Both corrections come from reading the London file rather than the data
-- dictionary, which is the order docs/DATA.md asks for: Inside Airbnb changes
-- its schema between snapshots, and a column it stopped publishing looks
-- exactly like a column full of legitimate values until something asserts
-- otherwise.
--
-- 1. Price is quoted in the listing's local currency, and London's is GBP --
--    each row's price quote carries "currency": "GBP" -- even though the
--    `price` column is printed with a dollar sign. A column named price_usd
--    holding pounds is a unit error waiting to be published in a report, so it
--    is renamed. The CHECK constraint follows the column automatically and
--    keeps its name, which carries no unit.
--
-- 2. `instant_bookable` is empty for every row of the snapshot (so are
--    `license` and `calendar_updated`; the scrape stopped populating them).
--    NOT NULL would reject every listing, and storing false for all of them
--    would tell the agent that nothing in London is instant-bookable, which is
--    a fact this project would be inventing. NULL is what "the snapshot does
--    not say" means, and a NULL satisfies no comparison, so a filter on it
--    returns nothing rather than something wrong. The column keeps its index
--    key slot so a snapshot that publishes the field again needs no migration,
--    and the amenity boolean budget 0001 documents is unchanged.

ALTER TABLE listings RENAME COLUMN price_usd TO price_gbp;

COMMENT ON COLUMN listings.price_gbp IS
    'Nightly price in GBP, the currency the London snapshot quotes, despite the '
    'dollar sign the source prints. NULL where the snapshot carries no price -- '
    'about a third of listings -- and a NULL satisfies no comparison, so a '
    'price filter excludes those listings rather than including them.';

ALTER TABLE listings ALTER COLUMN instant_bookable DROP NOT NULL;

COMMENT ON COLUMN listings.instant_bookable IS
    'NULL where the snapshot does not publish the field, which is every row of '
    'the 2026-06-19 London snapshot. Kept as a column so a snapshot that '
    'publishes it again is a reload rather than a migration.';

-- ------------------------------------------------------------- provenance --
-- Which snapshot the rows in this database came from.
--
-- Listing IDs are not stable between Inside Airbnb releases, so a retrieval
-- number or a relevance label without a snapshot date means nothing. Keeping
-- the date beside the data rather than only in the environment means a database
-- can always say what it holds, even when it outlives the shell that loaded it.

CREATE TABLE data_snapshot (
    -- One row, enforced: a second snapshot in the same database would make
    -- every listing id ambiguous.
    id            bool        PRIMARY KEY DEFAULT true CHECK (id),
    snapshot_date date        NOT NULL,
    city          text        NOT NULL,
    listing_count int4        NOT NULL,
    review_count  int4        NOT NULL,
    loaded_at     timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE data_snapshot IS
    'Which Inside Airbnb release these rows came from. One row, rewritten by '
    'each load. Carries no personal data: a date, a city, and two counts.';
