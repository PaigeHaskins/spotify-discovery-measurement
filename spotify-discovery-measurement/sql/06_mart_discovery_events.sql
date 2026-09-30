-- ============================================================================
-- 06_mart_discovery_events.sql  (MART LAYER)
-- ----------------------------------------------------------------------------
-- WHAT: One row per discovered artist with
--         (a) the OUTCOME: did you come back within the retention window?
--         (b) FEATURES known at the first encounter (for the model)
--         (c) the discovery PATH: active / passive / other (for causal work)
--         (d) an ELIGIBILITY flag that removes biased rows from analysis.
-- WHY:  This is the analysis-ready table: the model, the causal analysis, and
--       the dashboard all read from it.
--
-- THE TWO CENSORING PROBLEMS THIS TABLE SOLVES
--   burn_in  : first seen in the first {{BURN_IN_DAYS}} days of the export.
--              These are mostly artists you already knew before the export
--              began, so they are not real discoveries.
--   censored : first seen in the last {{RETENTION_WINDOW_DAYS}} days. We can't
--              see their full window yet, so labeling them "did not convert"
--              would be wrong. They count as discoveries but get no label.
--   eligible : everything else; the only rows used to measure conversion.
-- ============================================================================

DROP TABLE IF EXISTS mart_discovery_events;

CREATE TABLE mart_discovery_events AS
WITH bounds AS (
    -- The first and last day of the export (all listening, so the window
    -- matches the monthly KPI model exactly).
    SELECT MIN(listen_date) AS history_start,
           MAX(listen_date) AS history_end
    FROM stg_streams
),

discovery_day_depth AS (
    -- How deep was the first encounter? Streams, minutes, and distinct songs
    -- of this artist on the discovery day. A listener who plays three songs
    -- in a row is signaling more interest than one who hears a single track.
    SELECT
        f.artist_name,
        COUNT(*)                           AS streams_on_discovery_day,
        ROUND(SUM(s.ms_played) / 60000.0, 2) AS minutes_on_discovery_day,
        COUNT(DISTINCT s.track_name)       AS distinct_tracks_on_discovery_day
    FROM int_artist_first_listen AS f
    JOIN stg_foreground_streams AS s
      ON  s.artist_name  = f.artist_name
      AND s.listen_date  = f.first_listen_date
      AND s.is_qualified = 1
    GROUP BY f.artist_name
),

returns AS (
    -- The OUTCOME. Any qualified stream of the artist on a LATER day inside
    -- the window. Strictly later day (>) so a same-day binge is not counted
    -- as "coming back".
    SELECT
        f.artist_name,
        MIN(s.listen_date)            AS first_return_date,
        COUNT(DISTINCT s.listen_date) AS return_days_in_window
    FROM int_artist_first_listen AS f
    JOIN stg_foreground_streams AS s
      ON  s.artist_name  = f.artist_name
      AND s.is_qualified = 1
      AND s.listen_date  >  f.first_listen_date
      AND s.listen_date  <= DATE(f.first_listen_date, '+{{RETENTION_WINDOW_DAYS}} days')
    GROUP BY f.artist_name
)

SELECT
    f.*,
    d.streams_on_discovery_day,
    d.minutes_on_discovery_day,
    d.distinct_tracks_on_discovery_day,

    -- How the first stream started. See config.py for why only these two
    -- values are unambiguous enough for the causal comparison.
    CASE
        WHEN f.reason_start IN ({{ACTIVE_REASONS}})  THEN 'active'
        WHEN f.reason_start IN ({{PASSIVE_REASONS}}) THEN 'passive'
        ELSE 'other'
    END AS discovery_path,

    r.first_return_date,
    COALESCE(r.return_days_in_window, 0)                  AS return_days_in_window,
    CASE WHEN r.first_return_date IS NOT NULL THEN 1 ELSE 0 END AS converted,

    CASE
        WHEN f.first_listen_date <  DATE(b.history_start, '+{{BURN_IN_DAYS}} days')
            THEN 'burn_in'
        WHEN f.first_listen_date >  DATE(b.history_end, '-{{RETENTION_WINDOW_DAYS}} days')
            THEN 'censored'
        ELSE 'eligible'
    END AS eligibility
FROM int_artist_first_listen AS f
JOIN discovery_day_depth     AS d ON d.artist_name = f.artist_name
LEFT JOIN returns            AS r ON r.artist_name = f.artist_name
CROSS JOIN bounds            AS b;
