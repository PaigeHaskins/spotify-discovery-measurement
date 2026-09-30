"""
src/features.py
===============
Shared helpers that turn `mart_discovery_events` rows into numeric features.

WHY A SHARED MODULE
  The predictive model and the causal analysis both need numeric versions of
  the same columns. Building them in one place guarantees an "evening" or a
  "mobile" discovery means the same thing in both analyses.

TWO FEATURE SETS, ON PURPOSE
  * PRE_EXPOSURE features describe the situation BEFORE the first song
    finished: time of day, weekend, device, shuffle, offline. These are the
    only valid controls for the causal analysis, because they cannot be
    caused by how the song started.
  * ENGAGEMENT features describe what happened DURING the first encounter
    (listen length, how it ended, same-day depth). They are great predictors
    but they sit on the causal path (passive start -> shorter listen ->
    no return), so controlling for them would hide part of the effect we are
    trying to measure. The model uses them; the causal analysis does not.
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import config


def load_events(db_path: Path, eligible_only: bool = True) -> pd.DataFrame:
    """Read discovery events sorted by time (needed for the time-based split)."""
    with sqlite3.connect(db_path) as con:
        df = pd.read_sql("SELECT * FROM mart_discovery_events", con)
    if eligible_only:
        df = df[df["eligibility"] == "eligible"]
    return df.sort_values(["first_ts_utc", "first_stream_id"]).reset_index(drop=True)


def pre_exposure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Context known before/at the start of the first listen."""
    X = pd.DataFrame(index=df.index)

    # Time of day as four dayparts instead of a single number. Listening is
    # not smooth across the day (commute, workday, evening, late night), and
    # separate flags let the models treat each part of the day differently.
    # Each is also easy to check for balance in the causal analysis.
    h = df["hour_local"].astype(int)
    X["daypart_morning"] = h.between(5, 11).astype(int)
    X["daypart_afternoon"] = h.between(12, 16).astype(int)
    X["daypart_evening"] = h.between(17, 21).astype(int)
    X["daypart_night"] = ((h >= 22) | (h <= 4)).astype(int)
    X["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

    # Flags can be NULL (basic export or older rows). Fill with 0 and keep a
    # "missing" indicator so the model can tell "no" from "unknown".
    for col in ["shuffle", "offline"]:
        X[f"{col}_missing"] = df[col].isna().astype(int)
        X[col] = df[col].fillna(0).astype(int)

    # Device group as one-hot columns. A fixed list keeps the same columns in
    # every run, even if a device never appears in a given export.
    for level in ["mobile", "desktop_web", "speaker_tv_car", "unknown"]:
        X[f"platform_{level}"] = (df["platform_group"] == level).astype(int)

    return X


def time_trend(df: pd.DataFrame) -> pd.Series:
    """Months since the first eligible discovery (causal analysis only).

    It controls for drift in habits (new phone, new job, new app features)
    when comparing passive and active discoveries made around the same time.
    The PREDICTIVE model deliberately does not use it: with a time-based
    split, a trend learned on older years is extrapolated into the test
    years, which ranks new artists by date instead of by engagement.
    """
    t = pd.to_datetime(df["first_listen_date"])
    return (t - t.min()).dt.days / 30.44


def engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """What happened during the first encounter (model only; see docstring)."""
    X = pd.DataFrame(index=df.index)
    # Minutes of the first stream, capped at 10: a 25-minute live track should
    # not dominate. log1p compresses the long right tail.
    X["log_first_minutes"] = np.log1p((df["first_ms_played"] / 60000).clip(upper=10))
    X["log_streams_day0"] = np.log1p(df["streams_on_discovery_day"])
    X["distinct_tracks_day0"] = df["distinct_tracks_on_discovery_day"].clip(upper=10)
    # How the first stream ended: finished naturally vs skipped away.
    X["ended_trackdone"] = (df["reason_end"] == "trackdone").astype(int)
    X["ended_fwdbtn"] = (df["reason_end"] == "fwdbtn").astype(int)
    return X


def path_features(df: pd.DataFrame) -> pd.DataFrame:
    """How the first stream started, one-hot (model only)."""
    X = pd.DataFrame(index=df.index)
    # "active" (clicked the song) is the reference level, so each column reads
    # as "compared with clicking the song yourself". Leaving one level out
    # avoids perfectly redundant columns.
    X["path_passive"] = (df["discovery_path"] == "passive").astype(int)
    X["path_other"] = (df["discovery_path"] == "other").astype(int)
    return X


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """All features for the predictive model."""
    return pd.concat([pre_exposure_features(df), engagement_features(df), path_features(df)], axis=1)


# Human-readable names for the dashboard.
FEATURE_LABELS = {
    "daypart_morning": "Morning (5-11)",
    "daypart_afternoon": "Afternoon (12-4pm)",
    "daypart_evening": "Evening (5-9pm)",
    "daypart_night": "Late night (10pm-4am)",
    "platform_unknown": "Device not recorded",
    "is_weekend": "Weekend",
    "shuffle": "Shuffle on",
    "shuffle_missing": "Shuffle unknown",
    "offline": "Offline",
    "offline_missing": "Offline unknown",
    "platform_mobile": "Phone",
    "platform_desktop_web": "Desktop / web",
    "platform_speaker_tv_car": "Speaker / TV / car",
    "months_since_start": "Time trend (months)",
    "log_first_minutes": "Minutes of first stream",
    "log_streams_day0": "Streams on discovery day",
    "distinct_tracks_day0": "Songs heard on discovery day",
    "ended_trackdone": "First song played to the end",
    "ended_fwdbtn": "First song skipped away",
    "path_passive": "Autoplay start (vs. clicked)",
    "path_other": "Other start (vs. clicked)",
}
