"""
src/build_models.py
===================
STEP 2 OF THE PIPELINE: run the SQL files in sql/ in numeric order.

WHAT IT DOES
  Reads each .sql file, fills in {{PLACEHOLDERS}} from config.py, and executes
  it against the SQLite database. This is a tiny version of what dbt does:
  ordered models (staging -> intermediate -> marts), one definition per value.

WHY TEMPLATING INSTEAD OF HARD-CODED NUMBERS
  The 30-second threshold, the 30-day window, the burn-in and the background
  rules appear in several SQL files. Pulling them from config.py guarantees they always agree.
"""

import sqlite3
from pathlib import Path

import config


def sql_parameters() -> dict:
    """Values substituted into the SQL. Numbers are cast to int so a typo in
    config.py fails here, not silently inside SQL. Reason lists become a
    quoted, comma-separated list for SQL IN (...)."""
    def quoted(values):
        # Only simple lowercase words are allowed, which rules out SQL
        # injection through config values.
        for v in values:
            if not v.replace("_", "").replace("-", "").isalnum():
                raise ValueError(f"Unexpected reason_start value in config: {v!r}")
        return ", ".join(f"'{v}'" for v in values)

    def sql_string(value: str) -> str:
        # Standard SQL escaping: a single quote inside a string is doubled.
        return "'" + str(value).replace("'", "''") + "'"

    def name_list(names) -> str:
        # An empty IN () is invalid SQL. It must NOT become IN (NULL):
        # `x NOT IN (NULL)` is never true in SQL, which would silently empty
        # the table. An empty subquery is the safe "matches nothing" list.
        return ", ".join(sql_string(n) for n in names) if names else "SELECT NULL WHERE 0"

    # Keyword test built as one OR chain of LIKE clauses. The title text is
    # padded with spaces so " rain " matches the word but not "train".
    text = "(' ' || LOWER(COALESCE(s.album_name, '') || ' ' || COALESCE(s.track_name, '')) || ' ')"
    keyword_sql = " OR ".join(f"{text} LIKE {sql_string('%' + k.lower() + '%')}"
                              for k in config.BACKGROUND_KEYWORDS)
    night_start, night_end = config.BACKGROUND_NIGHT_START_HOURS

    return {
        "SESSION_GAP_MINUTES": str(int(config.SESSION_GAP_MINUTES)),
        "BACKGROUND_MAX_ACTION_RATE": str(float(config.BACKGROUND_MAX_ACTION_RATE)),
        "BACKGROUND_LONG_SESSION_MIN": str(int(config.BACKGROUND_LONG_SESSION_MIN)),
        "BACKGROUND_NIGHT_SESSION_MIN": str(int(config.BACKGROUND_NIGHT_SESSION_MIN)),
        "NIGHT_START_HOUR": str(int(night_start)),
        "NIGHT_END_HOUR": str(int(night_end)),
        "BACKGROUND_SESSION_SHARE": str(float(config.BACKGROUND_SESSION_SHARE)),
        "BACKGROUND_KEYWORD_SHARE": str(float(config.BACKGROUND_KEYWORD_SHARE)),
        "KEYWORD_MATCH": f"({keyword_sql})",
        "BACKGROUND_ALWAYS": name_list(config.BACKGROUND_ALWAYS),
        "BACKGROUND_NEVER": name_list(config.BACKGROUND_NEVER),
        "MIN_PLAY_MS": str(int(config.MIN_PLAY_MS)),
        "RETENTION_WINDOW_DAYS": str(int(config.RETENTION_WINDOW_DAYS)),
        "BURN_IN_DAYS": str(int(config.BURN_IN_DAYS)),
        "ACTIVE_REASONS": quoted(config.ACTIVE_REASONS),
        "PASSIVE_REASONS": quoted(config.PASSIVE_REASONS),
    }


def render(sql_text: str, params: dict) -> str:
    """Replace every {{NAME}} and fail loudly if any placeholder is left over."""
    for key, value in params.items():
        sql_text = sql_text.replace("{{" + key + "}}", value)
    if "{{" in sql_text:
        leftover = sql_text[sql_text.index("{{"): sql_text.index("{{") + 40]
        raise ValueError(f"Unfilled SQL placeholder near: {leftover}")
    return sql_text


def build(db_path: Path = config.DB_PATH, sql_dir: Path = config.SQL_DIR) -> list:
    """Execute all models in order; returns (model, row_count) pairs."""
    params = sql_parameters()
    sql_files = sorted(sql_dir.glob("*.sql"))
    results = []
    with sqlite3.connect(db_path) as con:
        for path in sql_files:
            con.executescript(render(path.read_text(encoding="utf-8"), params))
            # The table name is the file name without its number prefix,
            # e.g. 01_stg_streams.sql -> stg_streams.
            table = path.stem.split("_", 1)[1]
            n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            results.append((table, n))
            print(f"  built {table:<26} {n:>9,} rows")
    return results


if __name__ == "__main__":
    build()
