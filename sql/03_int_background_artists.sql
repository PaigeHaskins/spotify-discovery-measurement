-- ============================================================================
-- 03_int_background_artists.sql  (INTERMEDIATE LAYER)
-- ----------------------------------------------------------------------------
-- WHAT: The list of sleep / background artists, with the reason each one was
--       flagged. One row per flagged artist.
-- WHY:  Removing single plays is not enough for a discovery analysis: a
--       brown-noise "artist" played every night would still look like a
--       discovery that converted. Flagging at the ARTIST level removes them
--       from discovery entirely, while real artists who were only sometimes
--       played in the background keep their normal listening.
-- RULES (a flagged artist meets at least one)
--   sleep_or_noise_content : {{BACKGROUND_KEYWORD_SHARE}}+ of their streams have
--                            a sleep/noise phrase in the album or track title
--   background_sessions    : {{BACKGROUND_SESSION_SHARE}}+ of their streams were
--                            in background sessions (model 02)
--   manual                 : listed in config.BACKGROUND_ALWAYS
--   Artists in config.BACKGROUND_NEVER are never flagged.
-- ============================================================================

DROP TABLE IF EXISTS int_background_artists;

CREATE TABLE int_background_artists AS
WITH plays AS (
    SELECT
        s.artist_name,
        s.ms_played,
        ls.is_background_session,
        CASE WHEN {{KEYWORD_MATCH}} THEN 1 ELSE 0 END AS keyword_hit
    FROM stg_streams            AS s
    JOIN int_play_sessions      AS p  ON p.stream_id  = s.stream_id
    JOIN int_listening_sessions AS ls ON ls.session_id = p.session_id
    WHERE s.is_qualified = 1
),
per_artist AS (
    SELECT
        artist_name,
        COUNT(*)                              AS streams,
        ROUND(SUM(ms_played) / 3600000.0, 2)  AS hours,
        ROUND(AVG(keyword_hit), 3)            AS keyword_share,
        ROUND(AVG(is_background_session), 3)  AS background_session_share
    FROM plays
    GROUP BY artist_name
)
SELECT
    *,
    CASE
        WHEN artist_name IN ({{BACKGROUND_ALWAYS}})                THEN 'manual'
        WHEN keyword_share >= {{BACKGROUND_KEYWORD_SHARE}}          THEN 'sleep_or_noise_content'
        ELSE 'background_sessions'
    END AS background_reason
FROM per_artist
WHERE artist_name NOT IN ({{BACKGROUND_NEVER}})
  AND (   artist_name IN ({{BACKGROUND_ALWAYS}})
       OR keyword_share >= {{BACKGROUND_KEYWORD_SHARE}}
       OR background_session_share >= {{BACKGROUND_SESSION_SHARE}});

CREATE UNIQUE INDEX idx_bg_artist ON int_background_artists (artist_name);
