-- Runs once when the data directory is first created.
--
-- Both extensions go into the same database on purpose: Scout's retrieval
-- evaluation compares Brindle against pgvector over identical rows.

CREATE EXTENSION IF NOT EXISTS brindle;
CREATE EXTENSION IF NOT EXISTS vector;

-- Fail loudly here rather than three tasks later with a confusing error.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_am WHERE amname = 'brindle') THEN
        RAISE EXCEPTION 'brindle access method missing after CREATE EXTENSION';
    END IF;
    RAISE NOTICE 'brindle % and vector % ready',
        (SELECT extversion FROM pg_extension WHERE extname = 'brindle'),
        (SELECT extversion FROM pg_extension WHERE extname = 'vector');
END
$$;
