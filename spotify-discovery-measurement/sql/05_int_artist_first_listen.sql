-- ============================================================================
-- 05_int_artist_first_listen.sql  (INTERMEDIATE LAYER)
-- ----------------------------------------------------------------------------
-- WHAT: One row per artist: the first time you ever gave them a real (30s+)
--       stream, plus the context of that moment (how it started, device,
--       shuffle, time of day).
-- WHY:  This is the "exposure" event. Discovery Mode's question is what
--       happens AFTER a listener first meets an artist, so every discovery
--       metric starts from this row.
-- DESIGN CHOICES
--   * Only qualified streams count. A 5-second skip is not meeting an artist.
--   * Artists are identified by name. The export has no artist ID, so two
--     different artists with an identical name would be merged (rare; listed
--     as a limitation in the README).
--   * ROW_NUMBER instead of MIN(ts): MIN gives the time, but we also need the
--     other columns FROM that exact row, and stream_id breaks timestamp ties.
-- ============================================================================

DROP TABLE IF EXISTS int_artist_first_listen;

CREATE TABLE int_artist_first_listen AS
WITH ranked AS (
    SELECT
        s.*,
        ROW_NUMBER() OVER (
            PARTITION BY artist_name
            ORDER BY ts_utc, stream_id
        ) AS listen_rank
    FROM stg_foreground_streams AS s   -- sleep/background plays excluded
    WHERE is_qualified = 1
)
SELECT
    artist_name,
    stream_id        AS first_stream_id,
    ts_utc           AS first_ts_utc,
    ts_local         AS first_ts_local,
    listen_date      AS first_listen_date,
    track_name       AS first_track_name,
    ms_played        AS first_ms_played,
    hour_local,
    weekday,
    platform_group,
    reason_start,
    reason_end,
    shuffle,
    skipped,
    offline
FROM ranked
WHERE listen_rank = 1;

CREATE UNIQUE INDEX idx_first_artist ON int_artist_first_listen (artist_name);
