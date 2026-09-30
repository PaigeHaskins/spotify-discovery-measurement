"""
src/build_master.py
===================
Combine every Spotify Extended Streaming History JSON file into ONE master
data file: data/master/spotify_master.csv

WHY A MASTER FILE FIRST
  Spotify splits the history into dozens of JSON files (by year and size).
  One flat table is easier to explore, easy to open in Excel or Colab, and
  gives every later step a single, documented starting point.

WHAT IT KEEPS AND CHANGES
  * Keeps every row from every Audio and Video file, so nothing is lost
    (music, podcasts, audiobooks, video). Later steps filter to what they need.
  * Drops ip_addr: it identifies your home network and has no analytical use.
  * Removes exact duplicate rows (every field identical), which exports
    occasionally contain, and reports how many.
  * Adds helper columns: local time, date, content type, minutes played,
    a 30-second "stream" flag, and the source file each row came from.

RUN
    python -m src.build_master                      # reads data/raw/
    python -m src.build_master --raw path/to/folder
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


# Column order for the output: helper columns first, then Spotify's fields.
FRONT = ["ts_utc", "ts_local", "date_local", "year", "hour_local", "content_type",
         "artist_name", "track_name", "album_name", "episode_show_name", "episode_name",
         "audiobook_title", "audiobook_chapter_title", "ms_played", "minutes_played",
         "is_stream_30s", "reason_start", "reason_end", "shuffle", "skipped", "offline",
         "incognito_mode", "platform", "conn_country", "spotify_track_uri",
         "spotify_episode_uri", "audiobook_uri", "audiobook_chapter_uri",
         "offline_timestamp", "source_file"]


def read_all(raw_dir: Path) -> pd.DataFrame:
    """Read every Streaming_History_*.json file and tag rows with their file.

    rglob searches sub-folders, so the unzipped "Spotify Extended Streaming
    History" folder can sit anywhere inside data/raw/.
    """
    frames = []
    files = sorted(raw_dir.rglob("Streaming_History_*.json"))
    if not files:
        raise FileNotFoundError(f"No Streaming_History_*.json files under {raw_dir}")
    for path in files:
        rows = json.loads(path.read_text(encoding="utf-8"))
        df = pd.DataFrame(rows)
        df["source_file"] = path.name
        frames.append(df)
        print(f"  {path.name:<42} {len(df):>8,} rows")
    return pd.concat(frames, ignore_index=True)


def add_content_type(df: pd.DataFrame) -> pd.Series:
    """Label each row by what was played.

    Spotify fills a different URI depending on the content, so the URI that is
    present tells us the type. Rows with none (rare; usually local files or
    removed content) are labeled 'unknown' rather than guessed.
    """
    kind = np.select(
        [df["spotify_track_uri"].notna() | df["master_metadata_album_artist_name"].notna(),
         df["spotify_episode_uri"].notna() | df["episode_name"].notna(),
         df["audiobook_uri"].notna() | df["audiobook_title"].notna()],
        ["music", "podcast", "audiobook"], default="unknown")
    # Rows from the Video files are video content, whatever their URI says.
    return pd.Series(np.where(df["source_file"].str.contains("Video"), "video_" + kind, kind),
                     index=df.index)


def build(raw_dir: Path = config.RAW_DIR, tz: str = config.LOCAL_TIMEZONE) -> Path:
    df = read_all(raw_dir)
    n_read = len(df)

    # 1. Privacy: drop the IP address before anything is written.
    df = df.drop(columns=[c for c in ["ip_addr", "ip_addr_decrypted", "user_agent_decrypted",
                                      "username"] if c in df.columns])

    # 2. Exact duplicates (all fields identical except which file they came from).
    dedupe_cols = [c for c in df.columns if c != "source_file"]
    df = df.drop_duplicates(subset=dedupe_cols)
    n_dupes = n_read - len(df)

    # 3. Label the content type while Spotify's original column names are
    #    still in place, then give the long metadata columns friendlier names.
    df["content_type"] = add_content_type(df)

    # Friendlier names for the long Spotify metadata columns.
    df = df.rename(columns={"master_metadata_album_artist_name": "artist_name",
                            "master_metadata_track_name": "track_name",
                            "master_metadata_album_album_name": "album_name"})

    # 4. Time. `ts` is when playback STOPPED, in UTC. Local time is what
    #    matters for behavior ("late-night listening" means late in Boston).
    ts = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    local = ts.dt.tz_convert(tz)
    df["ts_utc"] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["ts_local"] = local.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["date_local"] = local.dt.strftime("%Y-%m-%d")
    df["year"] = local.dt.year
    df["hour_local"] = local.dt.hour

    # 5. Helper measures.
    df["minutes_played"] = (df["ms_played"] / 60000).round(2)
    # 30 seconds is the threshold at which a play counts as a stream.
    df["is_stream_30s"] = (df["ms_played"] >= config.MIN_PLAY_MS).astype(int)

    # 6. Order rows by time and columns logically, then save.
    df = df.sort_values(["ts_utc", "source_file"]).reset_index(drop=True)
    df = df[[c for c in FRONT if c in df.columns] + [c for c in df.columns if c not in FRONT and c != "ts"]]

    out = config.MASTER_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n  read {n_read:,} rows, removed {n_dupes:,} exact duplicates, "
          f"wrote {len(df):,} rows x {df.shape[1]} columns")
    print(f"  saved {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=config.RAW_DIR)
    build(parser.parse_args().raw)
