"""
src/checks.py
=============
STEP 3 OF THE PIPELINE: data-quality checks. The pipeline stops if any fail.

WHY
  A dashboard is only as trustworthy as the tables under it. Each check below
  recomputes a SQL result a SECOND, independent way (mostly in pandas) and
  confirms both answers agree. If a definition is ever edited incorrectly,
  the pipeline fails here instead of publishing a wrong number.
"""

import sqlite3
from pathlib import Path

import pandas as pd

import config


class DataQualityError(AssertionError):
    """Raised when a check fails; the message says which check and why."""


def _check(condition: bool, name: str, detail: str = "") -> str:
    if not condition:
        raise DataQualityError(f"FAILED: {name}. {detail}")
    return name


def run_checks(db_path: Path = config.DB_PATH) -> list:
    passed = []
    with sqlite3.connect(db_path) as con:
        raw = pd.read_sql("SELECT * FROM raw_streams", con)
        stg = pd.read_sql("SELECT * FROM stg_streams", con)
        fg = pd.read_sql("SELECT * FROM stg_foreground_streams", con)
        bg_artists = pd.read_sql("SELECT * FROM int_background_artists", con)
        bg_sessions = pd.read_sql(
            "SELECT p.stream_id FROM int_play_sessions p JOIN int_listening_sessions l "
            "USING (session_id) WHERE l.is_background_session = 1", con)
        first = pd.read_sql("SELECT * FROM int_artist_first_listen", con)
        events = pd.read_sql("SELECT * FROM mart_discovery_events", con)
        monthly = pd.read_sql("SELECT * FROM mart_monthly_kpis", con)

    # 1. Privacy: no personal identifiers made it into the database.
    pii = {"ip_addr", "ip_addr_decrypted", "user_agent_decrypted", "username", "conn_country"}
    passed.append(_check(not (pii & set(raw.columns)),
                         "no personal identifiers stored",
                         f"found {pii & set(raw.columns)}"))

    # 2. Staging keeps every raw play (no rows lost or duplicated by SQL).
    passed.append(_check(len(stg) == len(raw), "staging row count equals raw",
                         f"raw={len(raw)} stg={len(stg)}"))
    passed.append(_check(stg["stream_id"].is_unique, "stream_id is unique"))

    # 3. Qualified flag matches the threshold, checked in pandas.
    expected_q = (stg["ms_played"] >= config.MIN_PLAY_MS).astype(int)
    passed.append(_check((expected_q == stg["is_qualified"]).all(),
                         "qualified-stream flag matches threshold"))

    # 4. Background filter: foreground has no background artist and no play
    #    from a background session, and nothing else was dropped.
    passed.append(_check(not fg["artist_name"].isin(bg_artists["artist_name"]).any(),
                         "no background artist in foreground listening"))
    passed.append(_check(not fg["stream_id"].isin(bg_sessions["stream_id"]).any(),
                         "no background-session play in foreground listening"))
    expected_fg = stg[~stg["artist_name"].isin(bg_artists["artist_name"])
                      & ~stg["stream_id"].isin(bg_sessions["stream_id"])]
    passed.append(_check(len(expected_fg) == len(fg), "foreground = all plays minus background",
                         f"expected {len(expected_fg)}, got {len(fg)}"))
    passed.append(_check(not events["artist_name"].isin(bg_artists["artist_name"]).any(),
                         "no background artist counted as a discovery"))

    # 5. First listen really is the earliest qualified (foreground) stream per artist.
    q = fg[fg["is_qualified"] == 1].sort_values(["ts_utc", "stream_id"])
    expected_first = q.groupby("artist_name", as_index=False).first()[["artist_name", "stream_id"]]
    merged = first.merge(expected_first, on="artist_name", how="outer", indicator=True)
    passed.append(_check((merged["_merge"] == "both").all(),
                         "every streamed artist has exactly one first-listen row"))
    passed.append(_check((merged["first_stream_id"] == merged["stream_id"]).all(),
                         "first-listen row is the earliest qualified stream"))

    # 6. Conversion label recomputed in pandas from scratch.
    q_days = q[["artist_name", "listen_date"]].drop_duplicates()
    q_days = q_days.assign(listen_date=pd.to_datetime(q_days["listen_date"]))
    ev = events[["artist_name", "first_listen_date", "converted"]].copy()
    ev["first_listen_date"] = pd.to_datetime(ev["first_listen_date"])
    joined = ev.merge(q_days, on="artist_name")
    gap = (joined["listen_date"] - joined["first_listen_date"]).dt.days
    in_window = joined[(gap >= 1) & (gap <= config.RETENTION_WINDOW_DAYS)]
    expected_conv = ev["artist_name"].isin(in_window["artist_name"]).astype(int)
    mismatches = int((expected_conv.values != ev["converted"].values).sum())
    passed.append(_check(mismatches == 0, "conversion label matches independent recomputation",
                         f"{mismatches} artists disagree"))

    # 7. Return dates always fall inside the window.
    r = events.dropna(subset=["first_return_date"])
    gap = (pd.to_datetime(r["first_return_date"]) - pd.to_datetime(r["first_listen_date"])).dt.days
    passed.append(_check(gap.between(1, config.RETENTION_WINDOW_DAYS).all(),
                         "first return date is inside the retention window"))

    # 8. Eligibility respects both censoring rules.
    start = pd.to_datetime(stg["listen_date"]).min()
    end = pd.to_datetime(stg["listen_date"]).max()
    eligible = events[events["eligibility"] == "eligible"]
    d = pd.to_datetime(eligible["first_listen_date"])
    passed.append(_check(
        (d >= start + pd.Timedelta(days=config.BURN_IN_DAYS)).all()
        and (d <= end - pd.Timedelta(days=config.RETENTION_WINDOW_DAYS)).all(),
        "no eligible discovery in the burn-in or censored windows"))

    # 9. Monthly KPIs add back up to the staging totals.
    passed.append(_check(int(monthly["qualified_streams"].sum()) == int(fg["is_qualified"].sum()),
                         "monthly qualified streams sum to foreground total"))
    bg_hours = stg.loc[~stg["stream_id"].isin(fg["stream_id"]), "ms_played"].sum() / 3.6e6
    passed.append(_check(abs(monthly["background_hours"].sum() - bg_hours) < 0.01 * len(monthly) + 0.01,
                         "monthly background hours sum to set-aside total"))
    passed.append(_check(int(monthly["new_artists"].sum()) == int((events["eligibility"] != "burn_in").sum()),
                         "monthly new artists sum to non-burn-in discoveries"))
    passed.append(_check(int(monthly["converted_discoveries"].sum())
                         == int(events.loc[events["eligibility"] == "eligible", "converted"].sum()),
                         "monthly conversions sum to eligible conversions"))

    for name in passed:
        print(f"  ok  {name}")
    return passed


if __name__ == "__main__":
    run_checks()
