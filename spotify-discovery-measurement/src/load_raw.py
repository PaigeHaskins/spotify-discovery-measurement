"""
src/load_raw.py
===============
STEP 1 OF THE PIPELINE: raw Spotify JSON  ->  `raw_streams` table in SQLite.

WHAT IT DOES
  1. Finds your Spotify export files (either export type, see below).
  2. Maps each format onto ONE common set of columns.
  3. Drops personal identifiers (IP address, user agent, username).
  4. Converts UTC timestamps to your local time zone.
  5. Removes exact duplicate rows and non-music rows (podcasts, audiobooks).
  6. Writes the result to the `raw_streams` table.

WHY PYTHON HERE AND SQL AFTERWARDS
JSON parsing and time-zone conversion are easier in Python. Everything after
this step (sessionizing, discovery events, KPIs) is done in SQL, the same way
it would be done in BigQuery/dbt at Spotify.

THE TWO SPOTIFY EXPORT TYPES
  * Extended streaming history (recommended; takes up to 30 days to arrive)
      files: Streaming_History_Audio_*.json (older exports: endsong_*.json)
      has: reason_start, reason_end, shuffle, skipped, platform, track URI ...
  * Account data export (arrives in ~5 days, past year only)
      files: StreamingHistory_music_*.json (older: StreamingHistory0.json)
      has only: endTime, artistName, trackName, msPlayed
The project runs on either. With the basic export, analyses that need
reason_start (passive vs active) switch off and say so on the dashboard.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

import config

# Columns every downstream step can rely on, regardless of export type.
COMMON_COLUMNS = [
    "ts_utc", "ts_local", "artist_name", "track_name", "album_name",
    "track_uri", "ms_played", "platform", "reason_start", "reason_end",
    "shuffle", "skipped", "offline", "incognito_mode", "source_format",
]

# Fields that identify a person or device. We never keep them: they add no
# analytical value here, and the raw files should stay out of GitHub anyway.
PII_FIELDS = ["ip_addr", "ip_addr_decrypted", "user_agent_decrypted",
              "username", "conn_country"]


def find_export_files(raw_dir: Path) -> dict:
    """Return {'extended': [...], 'basic': [...]} lists of JSON files.

    rglob (recursive search) means you can unzip the Spotify download straight
    into data/raw/ and it will find the files inside the sub-folder.
    """
    extended, basic = [], []
    for path in sorted(raw_dir.rglob("*.json")):
        name = path.name.lower()
        # Video history and podcast-only files are not music listening.
        if "video" in name or "podcast" in name:
            continue
        if name.startswith("streaming_history_audio") or name.startswith("endsong"):
            extended.append(path)
        elif name.startswith("streaminghistory"):
            basic.append(path)
    return {"extended": extended, "basic": basic}


def _read_json_list(path: Path) -> list:
    """Read one export file. Each file is a JSON list of play records."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path.name} is not a list of play records; is it a Spotify streaming history file?")
    return data


def read_extended(files: list) -> pd.DataFrame:
    """Load extended-history files and rename columns to the common schema."""
    rows = []
    for path in files:
        rows.extend(_read_json_list(path))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    # Drop personal identifiers immediately, before anything is saved.
    df = df.drop(columns=[c for c in PII_FIELDS if c in df.columns])

    # Older exports may lack some optional fields; create them as empty so
    # the column mapping below never fails.
    for col in ["master_metadata_album_artist_name", "master_metadata_track_name",
                "master_metadata_album_album_name", "spotify_track_uri", "platform",
                "reason_start", "reason_end", "shuffle", "skipped", "offline",
                "incognito_mode"]:
        if col not in df.columns:
            df[col] = None

    out = pd.DataFrame({
        # NOTE: `ts` in the extended export is when the track STOPPED playing.
        "ts_utc": pd.to_datetime(df["ts"], utc=True, errors="coerce"),
        "artist_name": df["master_metadata_album_artist_name"],
        "track_name": df["master_metadata_track_name"],
        "album_name": df["master_metadata_album_album_name"],
        "track_uri": df["spotify_track_uri"],
        "ms_played": pd.to_numeric(df["ms_played"], errors="coerce"),
        "platform": df["platform"],
        "reason_start": df["reason_start"],
        "reason_end": df["reason_end"],
        "shuffle": df["shuffle"],
        "skipped": df["skipped"],
        "offline": df["offline"],
        "incognito_mode": df["incognito_mode"],
    })
    out["source_format"] = "extended"
    return out


def read_basic(files: list) -> pd.DataFrame:
    """Load account-data files; the fields they don't have are left empty."""
    rows = []
    for path in files:
        rows.extend(_read_json_list(path))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=COMMON_COLUMNS)

    out = pd.DataFrame({
        # endTime is also the moment the track stopped, in UTC ("YYYY-MM-DD HH:MM"),
        # so it lines up with the extended export's `ts`.
        "ts_utc": pd.to_datetime(df["endTime"], utc=True, errors="coerce"),
        "artist_name": df["artistName"],
        "track_name": df["trackName"],
        "ms_played": pd.to_numeric(df["msPlayed"], errors="coerce"),
    })
    for col in ["album_name", "track_uri", "platform", "reason_start", "reason_end",
                "shuffle", "skipped", "offline", "incognito_mode"]:
        out[col] = None
    out["source_format"] = "basic"
    return out


def read_master(master_path: Path) -> pd.DataFrame:
    """Load the master CSV built by src/build_master.py (music rows only).

    Why: the master file is the project's single documented starting point.
    It already has the IP address removed and exact duplicates dropped; this
    function only keeps music and renames columns to the common schema.
    """
    df = pd.read_csv(master_path, low_memory=False)
    df = df[df["content_type"] == "music"]
    out = pd.DataFrame({
        "ts_utc": pd.to_datetime(df["ts_utc"], utc=True, errors="coerce"),
        "artist_name": df["artist_name"],
        "track_name": df["track_name"],
        "album_name": df["album_name"],
        "track_uri": df["spotify_track_uri"],
        "ms_played": pd.to_numeric(df["ms_played"], errors="coerce"),
        "platform": df["platform"],
        "reason_start": df["reason_start"],
        "reason_end": df["reason_end"],
        "shuffle": df["shuffle"],
        "skipped": df["skipped"],
        "offline": df["offline"],
        "incognito_mode": df["incognito_mode"],
    })
    out["source_format"] = "extended"
    return out


def combine_sources(extended: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    """Merge both export types without double counting.

    Why: after a self-experiment you may request the quick account-data export
    to get recent days fast. It overlaps the extended history for the past
    year, so we only use basic rows AFTER the last extended timestamp.
    """
    if extended.empty:
        return basic
    if basic.empty:
        return extended
    cutoff = extended["ts_utc"].max()
    return pd.concat([extended, basic[basic["ts_utc"] > cutoff]], ignore_index=True)


def clean(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Filter to music plays, dedupe, add local time, and standardize types."""
    n_start = len(df)

    # 1. Keep music only. Podcast/audiobook rows have no artist name.
    df = df[df["artist_name"].notna() & (df["artist_name"].astype(str).str.strip() != "")]
    # 2. Rows with an unreadable timestamp or 0 ms played carry no information.
    df = df[df["ts_utc"].notna() & df["ms_played"].notna() & (df["ms_played"] > 0)]
    # 3. Spotify exports occasionally contain the exact same play twice.
    df = df.drop_duplicates(subset=["ts_utc", "artist_name", "track_name", "ms_played"])

    # 4. Local time. Stored as text 'YYYY-MM-DD HH:MM:SS' because SQLite has no
    #    datetime type and ISO text sorts correctly as a string.
    local = df["ts_utc"].dt.tz_convert(tz).dt.tz_localize(None)
    df = df.assign(
        ts_local=local.dt.strftime("%Y-%m-%d %H:%M:%S"),
        ts_utc=df["ts_utc"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # 5. True/False/None -> 1/0/NULL so SQL can SUM and AVG them.
    for col in ["shuffle", "skipped", "offline", "incognito_mode"]:
        df[col] = df[col].map({True: 1, False: 0, 1: 1, 0: 0}).astype("Int64")
    df["ms_played"] = df["ms_played"].astype("int64")

    df = df.sort_values(["ts_utc", "artist_name", "track_name"]).reset_index(drop=True)
    print(f"  cleaned {n_start:,} rows -> {len(df):,} music plays")
    return df[COMMON_COLUMNS]


def load(raw_dir: Path = config.RAW_DIR, db_path: Path = config.DB_PATH,
         tz: str = config.LOCAL_TIMEZONE, master_path: Path = None) -> dict:
    """Run the whole load step. Returns a small summary used later on.

    If a master CSV is given and exists, it is the source (normal real-data
    run). Otherwise the JSON files in raw_dir are read directly (sample mode
    and tests).
    """
    if master_path is not None and Path(master_path).exists():
        print(f"  reading master file {Path(master_path).name}")
        df = clean(read_master(master_path), tz)
        return _save(df, db_path, n_files=1)

    files = find_export_files(raw_dir)
    n_files = len(files["extended"]) + len(files["basic"])
    if n_files == 0:
        raise FileNotFoundError(
            f"No Spotify files found in {raw_dir}.\n"
            "Unzip your Spotify download into that folder. Expected files named like "
            "Streaming_History_Audio_2024_1.json or StreamingHistory_music_0.json.\n"
            "No data yet? Run:  python run_all.py --sample")
    print(f"  found {len(files['extended'])} extended + {len(files['basic'])} basic files")

    df = combine_sources(read_extended(files["extended"]), read_basic(files["basic"]))
    df = clean(df, tz)
    if df.empty:
        raise ValueError("The export files contained no music plays after cleaning.")

    return _save(df, db_path, n_files)


def _save(df: pd.DataFrame, db_path: Path, n_files: int) -> dict:
    """Write raw_streams to SQLite and summarize what was loaded."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as con:
        df.to_sql("raw_streams", con, if_exists="replace", index=False)

    # Share of plays that have reason_start tells later steps whether the
    # passive-vs-active analysis is possible.
    summary = {
        "n_files": n_files,
        "n_plays": int(len(df)),
        "has_reason_fields": bool(df["reason_start"].notna().mean() > 0.5),
        "formats": sorted(df["source_format"].unique().tolist()),
    }
    return summary


if __name__ == "__main__":
    print(load())
