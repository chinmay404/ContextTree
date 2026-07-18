-- external_files.data holds raw upload bytes and must be bytea. 001 created
-- it as jsonb by mistake, and 006's ADD COLUMN IF NOT EXISTS data bytea
-- no-opped because the column already existed. The frontend upload INSERT
-- sends a Buffer (binary-format parameter), which jsonb_recv rejects with
-- "unsupported jsonb version number 37" (0x25 = '%', first byte of %PDF) —
-- so no upload ever stored data here and the conversion drops nothing.

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name   = 'external_files'
          AND column_name  = 'data'
          AND data_type    = 'jsonb'
    ) THEN
        ALTER TABLE external_files
            ALTER COLUMN data TYPE bytea USING NULL;
    END IF;
END $$;
