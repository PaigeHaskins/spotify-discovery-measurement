-- ============================================================================
-- 04_stg_foreground_streams.sql  (STAGING LAYER, filtered)
-- ----------------------------------------------------------------------------
-- WHAT: Every play that counts as real, attentive listening:
--         * not in a background session, and
--         * not by a background artist.
-- WHY:  All discovery, model, causal, and experiment work reads from this
--       table, so the sleep/background rule is applied in exactly one place.
--       stg_streams keeps everything, so the set-aside listening can still be
--       reported on the dashboard.
-- ============================================================================

DROP TABLE IF EXISTS stg_foreground_streams;

CREATE TABLE stg_foreground_streams AS
SELECT s.*
FROM stg_streams            AS s
JOIN int_play_sessions      AS p  ON p.stream_id  = s.stream_id
JOIN int_listening_sessions AS ls ON ls.session_id = p.session_id
WHERE ls.is_background_session = 0
  AND s.artist_name NOT IN (SELECT artist_name FROM int_background_artists);

CREATE INDEX idx_fg_artist_date ON stg_foreground_streams (artist_name, listen_date);
CREATE INDEX idx_fg_date ON stg_foreground_streams (listen_date);
CREATE UNIQUE INDEX idx_fg_stream ON stg_foreground_streams (stream_id);
