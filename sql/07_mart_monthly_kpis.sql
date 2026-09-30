-- ============================================================================
-- 07_mart_monthly_kpis.sql  (MART LAYER)
-- ----------------------------------------------------------------------------
-- WHAT: One row per month with the program-health KPIs shown on the
--       dashboard: volume, short-play (skip) rate, breadth, discovery rate,
--       and 30-day conversion of new artists.
-- WHY:  The posting asks for dashboards that "track model health, customer
--       metrics, and program performance". This is that table: a small,
--       stable set of KPIs with the denominators kept alongside the rates,
--       so nobody reads a rate built on 3 events as a trend.
-- QUALITY FLAGS (the dashboard hides flagged months instead of plotting noise)
--   discovery_measurable = 0 : month starts inside the burn-in window
--   is_partial_month     = 1 : export starts or ends mid-month
-- ============================================================================

DROP TABLE IF EXISTS mart_monthly_kpis;

CREATE TABLE mart_monthly_kpis AS
WITH bounds AS (
    SELECT MIN(listen_date) AS history_start,
           MAX(listen_date) AS history_end,
           DATE(MIN(listen_date), '+{{BURN_IN_DAYS}} days') AS burn_in_end
    FROM stg_streams
),

background AS (
    -- Set-aside listening per month, reported next to the foreground KPIs.
    SELECT strftime('%Y-%m', s.listen_date) AS month,
           ROUND(SUM(s.ms_played) / 3600000.0, 2) AS background_hours
    FROM stg_streams AS s
    WHERE s.stream_id NOT IN (SELECT stream_id FROM stg_foreground_streams)
    GROUP BY 1
),

listening AS (
    -- Consumption KPIs from every play.
    SELECT
        strftime('%Y-%m', listen_date)           AS month,
        COUNT(*)                                 AS plays,
        SUM(is_qualified)                        AS qualified_streams,
        ROUND(SUM(ms_played) / 3600000.0, 2)     AS listening_hours,
        -- Share of plays under the stream threshold: a consistent skip proxy
        -- that works for both export types (the `skipped` flag is often NULL).
        ROUND(1.0 - AVG(is_qualified), 4)        AS short_play_rate,
        COUNT(DISTINCT CASE WHEN is_qualified = 1 THEN artist_name END) AS distinct_artists
    FROM stg_foreground_streams          -- attentive listening only
    GROUP BY 1
),

discovery AS (
    -- Discovery KPIs, dated by the month of the first encounter.
    SELECT
        strftime('%Y-%m', first_listen_date) AS month,
        SUM(CASE WHEN eligibility <> 'burn_in'  THEN 1 ELSE 0 END)         AS new_artists,
        SUM(CASE WHEN eligibility =  'eligible' THEN 1 ELSE 0 END)         AS eligible_discoveries,
        SUM(CASE WHEN eligibility =  'eligible' THEN converted ELSE 0 END) AS converted_discoveries
    FROM mart_discovery_events
    GROUP BY 1
)

SELECT
    l.month,
    DATE(l.month || '-01') AS month_start,
    l.plays,
    l.qualified_streams,
    l.listening_hours,
    l.short_play_rate,
    l.distinct_artists,
    COALESCE(bg.background_hours, 0) AS background_hours,
    COALESCE(d.new_artists, 0) AS new_artists,
    -- Normalizing by volume separates "discovering more" from "listening more".
    ROUND(100.0 * COALESCE(d.new_artists, 0) / NULLIF(l.qualified_streams, 0), 3)
        AS new_artists_per_100_streams,
    COALESCE(d.eligible_discoveries, 0)  AS eligible_discoveries,
    COALESCE(d.converted_discoveries, 0) AS converted_discoveries,
    ROUND(1.0 * d.converted_discoveries / NULLIF(d.eligible_discoveries, 0), 4)
        AS conversion_rate,
    CASE WHEN DATE(l.month || '-01') >= b.burn_in_end THEN 1 ELSE 0 END
        AS discovery_measurable,
    CASE WHEN DATE(l.month || '-01') < b.history_start
           OR DATE(l.month || '-01', '+1 month', '-1 day') > b.history_end
         THEN 1 ELSE 0 END
        AS is_partial_month
FROM listening AS l
LEFT JOIN discovery AS d ON d.month = l.month
LEFT JOIN background AS bg ON bg.month = l.month
CROSS JOIN bounds  AS b
ORDER BY l.month;
