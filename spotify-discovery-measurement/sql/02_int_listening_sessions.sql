-- ============================================================================
-- 02_int_listening_sessions.sql  (INTERMEDIATE LAYER)
-- ----------------------------------------------------------------------------
-- WHAT: Groups plays into listening sessions and flags BACKGROUND sessions:
--       long stretches of autoplay where I almost never skipped or clicked.
--       Builds two tables:
--         int_play_sessions      one row per play -> its session_id
--         int_listening_sessions one row per session, with its stats and flag
-- WHY:  Sleep sounds and all-night background playlists are not listening.
--       The clearest fingerprint of "not paying attention" is behavior, not
--       genre: awake listening is full of skips and clicks, while a speaker
--       playing jazz all night runs for hours with no one touching it.
-- RULE: a background session has an action rate (skips/clicks after the first
--       track) of at most {{BACKGROUND_MAX_ACTION_RATE}} AND either
--         * runs {{BACKGROUND_LONG_SESSION_MIN}}+ minutes at any time, or
--         * starts at night and runs {{BACKGROUND_NIGHT_SESSION_MIN}}+ minutes.
-- ============================================================================

DROP TABLE IF EXISTS int_play_sessions;

CREATE TABLE int_play_sessions AS
WITH timed AS (
    SELECT
        stream_id,
        -- `ts` marks when a play ENDED, so its start is end minus duration.
        -- julianday() gives fractional days, which subtract cleanly.
        julianday(ts_utc)                              AS end_day,
        julianday(ts_utc) - ms_played / 86400000.0     AS start_day,
        -- Any sign of a person: clicking a song, pressing play, or skipping.
        CASE WHEN reason_start IN ('clickrow', 'playbtn', 'fwdbtn', 'backbtn')
               OR reason_end   IN ('fwdbtn', 'backbtn')
             THEN 1 ELSE 0 END                          AS user_action
    FROM stg_streams
),
gaps AS (
    -- Minutes of silence since the previous play ended.
    SELECT *,
           (start_day - LAG(end_day) OVER (ORDER BY stream_id)) * 1440 AS gap_minutes
    FROM timed
)
SELECT
    stream_id,
    user_action,
    -- A running count of "new session" markers gives every play a session id.
    SUM(CASE WHEN gap_minutes IS NULL OR gap_minutes > {{SESSION_GAP_MINUTES}} THEN 1 ELSE 0 END)
        OVER (ORDER BY stream_id ROWS UNBOUNDED PRECEDING) AS session_id
FROM gaps;

CREATE INDEX idx_play_sessions ON int_play_sessions (stream_id);

DROP TABLE IF EXISTS int_listening_sessions;

CREATE TABLE int_listening_sessions AS
WITH plays AS (
    SELECT
        p.session_id,
        p.user_action,
        s.ms_played,
        s.hour_local,
        s.ts_local,
        ROW_NUMBER() OVER (PARTITION BY p.session_id ORDER BY p.stream_id) AS position
    FROM int_play_sessions AS p
    JOIN stg_streams      AS s ON s.stream_id = p.stream_id
),
stats AS (
    SELECT
        session_id,
        COUNT(*)                                             AS n_plays,
        MIN(ts_local)                                        AS first_play_end_local,
        MAX(CASE WHEN position = 1 THEN hour_local END)      AS start_hour,
        ROUND(SUM(ms_played) / 60000.0, 1)                   AS minutes,
        -- The first track is always started by someone, so it doesn't count.
        SUM(CASE WHEN position > 1 THEN user_action ELSE 0 END) AS actions_after_start
    FROM plays
    GROUP BY session_id
)
SELECT
    *,
    ROUND(1.0 * actions_after_start / MAX(n_plays - 1, 1), 4) AS action_rate,
    CASE
        WHEN 1.0 * actions_after_start / MAX(n_plays - 1, 1) <= {{BACKGROUND_MAX_ACTION_RATE}}
         AND (minutes >= {{BACKGROUND_LONG_SESSION_MIN}}
              OR ((start_hour >= {{NIGHT_START_HOUR}} OR start_hour < {{NIGHT_END_HOUR}})
                  AND minutes >= {{BACKGROUND_NIGHT_SESSION_MIN}}))
        THEN 1 ELSE 0
    END AS is_background_session
FROM stats;

CREATE UNIQUE INDEX idx_sessions ON int_listening_sessions (session_id);
