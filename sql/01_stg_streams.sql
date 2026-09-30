-- ============================================================================
-- 01_stg_streams.sql  (STAGING LAYER)
-- ----------------------------------------------------------------------------
-- WHAT: One row per music play, with typed columns and derived fields every
--       later model needs (calendar date, hour, qualified-stream flag, device
--       group).
-- WHY:  A staging model is the single cleaned source of truth. Every metric
--       downstream reads from here, so a definition (e.g. "a stream is 30s+")
--       is written exactly once.
-- NOTE: Written in SQLite. {{MIN_PLAY_MS}} is filled in from config.py by
--       src/build_models.py. Porting to BigQuery/dbt: swap strftime() for
--       EXTRACT()/FORMAT_DATE() and the placeholders for dbt Jinja variables;
--       the logic is identical.
-- ============================================================================

DROP TABLE IF EXISTS stg_streams;

CREATE TABLE stg_streams AS
SELECT
    -- Stable surrogate key. Ordering by time then name makes it deterministic,
    -- which we need to break ties when two plays share a timestamp.
    ROW_NUMBER() OVER (ORDER BY ts_utc, artist_name, track_name) AS stream_id,

    ts_utc,
    ts_local,
    DATE(ts_local)                                   AS listen_date,
    CAST(strftime('%H', ts_local) AS INTEGER)        AS hour_local,
    CAST(strftime('%w', ts_local) AS INTEGER)        AS weekday,   -- 0 = Sunday

    artist_name,
    track_name,
    album_name,
    track_uri,
    ms_played,

    -- A play only counts as a real stream after the minimum listen time.
    -- Shorter plays are treated as skips (a "short play").
    CASE WHEN ms_played >= {{MIN_PLAY_MS}} THEN 1 ELSE 0 END AS is_qualified,

    -- Spotify's platform strings are messy ("Android OS 13 API 33 (Google,
    -- Pixel 7)", "windows", "Partner sonos ..."). Bucketing them gives a
    -- feature with a few stable levels instead of hundreds of rare ones.
    CASE
        WHEN platform IS NULL
          OR LOWER(platform) IN ('not_applicable', 'unknown')    THEN 'unknown'
        -- Partner devices first: strings like "Partner android_tv ..." would
        -- otherwise be misread as a phone.
        WHEN LOWER(platform) LIKE '%partner%'
          OR LOWER(platform) LIKE '%cast%'                      THEN 'speaker_tv_car'
        WHEN LOWER(platform) LIKE '%android%'
          OR LOWER(platform) LIKE '%ios%'
          OR LOWER(platform) LIKE '%iphone%'                    THEN 'mobile'
        WHEN LOWER(platform) LIKE '%windows%'
          OR LOWER(platform) LIKE '%os x%'
          OR LOWER(platform) LIKE '%osx%'
          OR LOWER(platform) LIKE '%mac%'
          OR LOWER(platform) LIKE '%linux%'
          OR LOWER(platform) LIKE '%web%'                       THEN 'desktop_web'
        ELSE 'speaker_tv_car'   -- Sonos, cast, TVs, cars, game consoles
    END AS platform_group,

    COALESCE(reason_start, 'unknown') AS reason_start,
    COALESCE(reason_end,   'unknown') AS reason_end,
    shuffle,
    skipped,
    offline,
    incognito_mode,
    source_format
FROM raw_streams;

-- Indexes make the self-joins in the next models fast on years of history.
CREATE INDEX idx_stg_artist_date ON stg_streams (artist_name, listen_date);
CREATE INDEX idx_stg_date ON stg_streams (listen_date);
