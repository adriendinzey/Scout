-- The amenities that earn an index key column, and the lookup row that says so.
--
-- Chosen from real frequency counts over the 2026-06-19 London snapshot
-- (92,633 listings, 8,735 distinct amenity spellings), confirmed before this
-- migration was written, because the set is baked into the index: adding a
-- column later means rebuilding it and re-running anything measured on it.
--
-- Index key budget. PostgreSQL allows 32 key columns per index. 0001 spends 17
-- (the embedding plus 16 filter columns), leaving 15 for amenities. 13 are
-- spent here and 2 are deliberately left unspent.
--
-- Why these 13. The criterion is selectivity against demand, not popularity. A
-- predicate applied *after* the ranked scan keeps roughly its prevalence share
-- of the candidate pool, so a near-universal amenity costs almost nothing as an
-- array post-filter -- wifi is in 97.1% of listings, so checking it afterwards
-- drops 3% of candidates -- while a selective one collapses the pool: pets
-- allowed is in 13.3%, which turns 100 candidates into about 13 rows. The
-- columns therefore go to the selective amenities guests actually ask for, and
-- the universal ones (wifi, kitchen, washer, heating, TV) stay in
-- `listings.amenities`, where the retrieval layer applies them as a post-filter
-- and records that it did.
--
-- Prevalence in the snapshot this was chosen on, so a later snapshot's drift is
-- visible: workspace 44.5%, self check-in 40.8%, dryer 36.9%, bathtub 36.9%,
-- dishwasher 36.2%, outdoor space 31.6%, private entrance 24.8%, free parking
-- 20.8%, crib 18.5%, elevator 18.4%, pets 13.3%, air conditioning 11.8%,
-- gym 7.7%.
--
-- NOT NULL DEFAULT false, not nullable. These are derived from a list that is
-- always present, so "absent" means false; a NULL would satisfy no comparison
-- and an amenity filter would then silently exclude listings instead of
-- matching false against them. The retrieval layer already treats these columns
-- as always-known.
--
-- The values are not backfilled here. Which spellings mean which amenity is
-- decided by the pattern table in the amenity module -- "Free dryer -- In unit"
-- and "Dryer" are one appliance, "Hair dryer" is not -- and a second copy of
-- those rules in SQL would be a second thing to keep in step, whose only
-- symptom when it drifted would be a listing that quietly stopped matching an
-- amenity it has. So the columns are created false and filled by
-- `scout data amenities --backfill`, which every `scout data load` also runs as
-- part of the same transaction that writes the arrays they derive from.

ALTER TABLE listings
    ADD COLUMN has_dedicated_workspace bool NOT NULL DEFAULT false,
    ADD COLUMN has_self_checkin        bool NOT NULL DEFAULT false,
    ADD COLUMN has_bathtub             bool NOT NULL DEFAULT false,
    ADD COLUMN has_dryer               bool NOT NULL DEFAULT false,
    ADD COLUMN has_dishwasher          bool NOT NULL DEFAULT false,
    ADD COLUMN has_outdoor_space       bool NOT NULL DEFAULT false,
    ADD COLUMN has_free_parking        bool NOT NULL DEFAULT false,
    ADD COLUMN has_private_entrance    bool NOT NULL DEFAULT false,
    ADD COLUMN has_elevator            bool NOT NULL DEFAULT false,
    ADD COLUMN has_crib                bool NOT NULL DEFAULT false,
    ADD COLUMN has_pets_allowed        bool NOT NULL DEFAULT false,
    ADD COLUMN has_air_conditioning    bool NOT NULL DEFAULT false,
    ADD COLUMN has_gym                 bool NOT NULL DEFAULT false;

-- What each column counts, where the boundary is not obvious. A merge that
-- looks harmless is how a column becomes a lie: "Free parking on premises" and
-- "Paid parking off premises" are not the same answer to a guest who asked
-- about parking, and "Hair dryer" is not a tumble dryer.

COMMENT ON COLUMN listings.has_dedicated_workspace IS
    'A desk or work area the listing advertises. The flag the "good for working '
    'remotely" class of query turns on.';
COMMENT ON COLUMN listings.has_self_checkin IS
    'The listing advertises self check-in. The lock hardware the source lists '
    'beside it (lockbox, keypad, smart lock) is not counted: it adds one '
    'listing in the snapshot, and "Building staff" or "Doorman" is a person '
    'handing over a key, which is the opposite of checking yourself in.';
COMMENT ON COLUMN listings.has_bathtub IS
    'A bath, the fixture. The numeric bathrooms column counts rooms without '
    'saying what is in them.';
COMMENT ON COLUMN listings.has_dryer IS
    'A clothes dryer, free or paid, in the unit or in the building -- the '
    'question is whether clothes can be dried. Not a hair dryer, and not a '
    'laundromat down the road. 99.6% of listings with one also have a washer, '
    'so this column carries the laundry pair''s selectivity.';
COMMENT ON COLUMN listings.has_dishwasher IS 'A dishwasher. Never the clothes washer.';
COMMENT ON COLUMN listings.has_outdoor_space IS
    'Private, shared or unqualified patio, balcony, terrace or backyard: one '
    'column for "is there outside space", because the source splits it across '
    'four words and a guest asking for a garden does not care which was used. '
    'A view is not space -- "Garden view" and "Courtyard view" are excluded.';
COMMENT ON COLUMN listings.has_free_parking IS
    'Parking at the property that costs nothing: driveway, garage, carport or '
    'lot, on premises. Every paid spelling is excluded. "Free street parking" '
    'is excluded too -- it describes the street rather than a space the '
    'property can promise -- and stays available as an array post-filter; '
    'counting it here would move the column from 20.8% to 25.8% of listings.';
COMMENT ON COLUMN listings.has_private_entrance IS
    'The usual proxy for privacy in a shared building.';
COMMENT ON COLUMN listings.has_elevator IS
    'A lift. This snapshot publishes no accessibility fields, so it is the only '
    'structural signal for a guest who cannot manage stairs.';
COMMENT ON COLUMN listings.has_crib IS
    'Somewhere for a baby to sleep: a cot or a travel cot, including one that '
    'is paid for or has to be requested -- the alternative to requesting it is '
    'nothing to sleep in.';
COMMENT ON COLUMN listings.has_pets_allowed IS
    'Pets are permitted. A hard requirement, and the most selective column '
    'here that guests ask for by name.';
COMMENT ON COLUMN listings.has_air_conditioning IS
    'Air conditioning of any kind, including portable, window and split units. '
    'Fans are excluded: "Portable fans" is in 19% of listings, and answering a '
    'question about cooling with a desk fan is the merge this avoids.';
COMMENT ON COLUMN listings.has_gym IS
    'Exercise equipment at the property or in its building. "Gym nearby" is '
    'excluded -- a gym the guest has to find and pay for is not the amenity '
    'being asked about.';

-- ------------------------------------------------------- the lookup table --
-- Which boolean column, if any, serves each amenity spelling.
--
-- 0001 gave `amenities` an `is_indexed` flag, which answers "does this amenity
-- have a column" but not "which one", and the retrieval layer needs the second:
-- it maps an amenity *name* to the column that filters it, so a request for
-- "Pocket wifi" or "Free dryer -- In unit" lands on the same column as the
-- plain spelling. One text column answers both questions, and `is_indexed`
-- becomes generated from it so the two can never disagree.

ALTER TABLE amenities ADD COLUMN indexed_column text;

-- A lookup row pointing at a column that does not exist would surface either as
-- a failed CREATE INDEX much later or as a query against a missing column, so
-- the set of legal values is pinned here. A migration that adds an amenity
-- column updates this constraint in the same file, which also makes an amenity
-- group added in code without a migration fail loudly at backfill time rather
-- than at index build.
ALTER TABLE amenities ADD CONSTRAINT amenities_indexed_column_exists CHECK (
    indexed_column IS NULL OR indexed_column IN (
        'has_dedicated_workspace',
        'has_self_checkin',
        'has_bathtub',
        'has_dryer',
        'has_dishwasher',
        'has_outdoor_space',
        'has_free_parking',
        'has_private_entrance',
        'has_elevator',
        'has_crib',
        'has_pets_allowed',
        'has_air_conditioning',
        'has_gym'
    )
);

-- Dropped and re-added as generated rather than left as a plain flag: two
-- hand-maintained columns that have to agree eventually do not, and the symptom
-- would be an amenity the index can filter that nothing knows to filter with.
--
-- Dropping it discards whatever it held, and the old column recorded a choice
-- that cannot be recovered from the new one: it said *that* an amenity was
-- indexed, never which column filtered it, and no such columns existed before
-- this migration. So rather than claiming no database has a flag set, this
-- refuses to run on one that does -- a migration cannot be undone once the
-- runner has recorded it, and a lost selection would surface as an index built
-- on the wrong columns.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM amenities WHERE is_indexed) THEN
        RAISE EXCEPTION
            'amenities.is_indexed is set on % row(s); this migration replaces the '
            'column with one generated from indexed_column and cannot carry those '
            'values over, because the old flag does not say which column filters '
            'the amenity. Record the selection, clear the flag, and re-run.',
            (SELECT count(*) FROM amenities WHERE is_indexed);
    END IF;
END $$;

ALTER TABLE amenities DROP COLUMN is_indexed;
ALTER TABLE amenities ADD COLUMN is_indexed bool NOT NULL
    GENERATED ALWAYS AS (indexed_column IS NOT NULL) STORED;

COMMENT ON COLUMN amenities.indexed_column IS
    'The listings boolean column that filters this amenity, or NULL when the '
    'only way to filter it is the amenities array. Several spellings share one '
    'column: the 13 columns cover 141 of the snapshot''s 8,735 spellings, '
    'which is where most of the grouping work went -- 36 ways to say free '
    'parking on premises, 60 to say gym.';
COMMENT ON COLUMN amenities.is_indexed IS
    'Generated: whether this amenity has an indexed boolean column. Read by the '
    'index build to assemble its key column list.';
