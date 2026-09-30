"""
src/sample_data.py
==================
Generates a FAKE Spotify extended streaming history in the exact export
format, so the project can run end to end without anyone's personal data.

WHY THIS EXISTS
  1. Anyone who clones the repo can run it:  python run_all.py --sample
  2. Testing with known answers. The generator PLANTS effects whose true
     direction we know, so the tests can check the pipeline recovers them:
       * passive discoveries convert less (true effect built in), and they
         also happen more on speakers and late at night, which ALSO lower
         conversion. That is deliberate confounding: the naive comparison
         should look worse than the matched one.
       * engaged first listens (full play, several songs that day) predict
         conversion, so the model should beat the baseline.
  3. It deliberately includes the mess real exports have: podcasts, exact
     duplicate rows, sub-30-second skips, IP addresses, and daylight-saving
     time changes.
"""

import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PLATFORMS = {
    "mobile": "Android OS 14 API 34 (Google, Pixel 8)",
    "desktop_web": "windows",
    "speaker_tv_car": "Partner sonos_arc Sonos Arc",
}
# Typical listening hours per device (local time).
HOURS = {"mobile": [7, 8, 12, 17, 18, 21], "desktop_web": list(range(9, 18)),
         "speaker_tv_car": [18, 19, 20, 21, 22, 23]}

# Planted "truth" for conversion (log-odds scale).
BASE_LOGIT = -0.2
PASSIVE_EFFECT = -0.9     # the causal effect the analysis should detect
SPEAKER_EFFECT = -0.6     # confounder: speakers get more passive discoveries
LATE_EFFECT = -0.5        # confounder: late-night listening does too


def _sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def generate(out_dir: Path, seed: int = 3, start: str = "2024-09-01", n_days: int = 560) -> Path:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A library of artists you already know, with a few favorites dominating
    # (Zipf-like weights), which is what real listening looks like.
    known = [f"Known Artist {i:03d}" for i in range(300)]
    weights = 1 / np.arange(1, len(known) + 1)
    weights /= weights.sum()

    new_count = 0
    scheduled = defaultdict(list)   # day index -> artists due for a return play
    records = []
    day0 = pd.Timestamp(start)

    def play(ts_start, artist, track, platform_key, shuffle, reason_start, full=None):
        """Create one play record; returns the time the play ended."""
        if full is None:
            full = rng.random() > 0.2
        full = bool(full)   # numpy booleans are not JSON-serializable
        ms = int(rng.integers(120_000, 280_000)) if full else int(rng.integers(1_000, 29_000))
        end = ts_start + pd.Timedelta(milliseconds=ms)
        records.append({
            "ts_local": end, "platform": PLATFORMS[platform_key], "ms_played": ms,
            "conn_country": "US", "ip_addr": "203.0.113.7",   # PII the loader must drop
            "master_metadata_track_name": track,
            "master_metadata_album_artist_name": artist,
            "master_metadata_album_album_name": f"{artist} - Album",
            "spotify_track_uri": f"spotify:track:{zlib.crc32(f'{artist}|{track}'.encode())}",
            "episode_name": None, "episode_show_name": None, "spotify_episode_uri": None,
            "reason_start": reason_start,
            "reason_end": "trackdone" if full else "fwdbtn",
            "shuffle": shuffle, "skipped": not full, "offline": bool(rng.random() < 0.05),
            "offline_timestamp": None, "incognito_mode": False,
        })
        return end + pd.Timedelta(seconds=int(rng.integers(1, 20)))

    for d in range(n_days):
        day = day0 + pd.Timedelta(days=d)
        for _ in range(int(rng.poisson(3)) + 1):                     # sessions today
            pkey = rng.choice(list(PLATFORMS), p=[0.55, 0.25, 0.20])
            hour = int(rng.choice(HOURS[pkey]))
            t = day + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
            shuffle = bool(rng.random() < 0.5)
            for _ in range(int(rng.poisson(14)) + 3):                # plays in session
                if scheduled[d] and rng.random() < 0.5:
                    # A return visit to an artist discovered earlier.
                    artist = scheduled[d].pop()
                    t = play(t, artist, f"{artist} Song {rng.integers(1, 9)}", pkey, shuffle,
                             "clickrow" if rng.random() < 0.6 else "trackdone", full=rng.random() < 0.9)
                elif rng.random() < 0.035:
                    # A brand-new artist: the discovery event.
                    new_count += 1
                    artist = f"New Artist {new_count:05d}"
                    late = hour >= 21
                    speaker = pkey == "speaker_tv_car"
                    p_passive = 0.75 if (speaker or late) else 0.45    # confounding
                    passive = rng.random() < p_passive
                    converts = rng.random() < _sigmoid(BASE_LOGIT + PASSIVE_EFFECT * passive
                                                       + SPEAKER_EFFECT * speaker + LATE_EFFECT * late)
                    # 10% of the time the artist is first skipped in under 30s:
                    # that must NOT count as the discovery.
                    if rng.random() < 0.10:
                        t = play(t, artist, f"{artist} Song 0", pkey, shuffle, "trackdone", full=False)
                    # Engagement depends on whether you'll come back: this is
                    # the signal the predictive model should find.
                    full = rng.random() < (0.9 if converts else 0.55)
                    first_ms_full = full
                    t_start = t
                    t = play(t_start, artist, f"{artist} Song 1", pkey, shuffle,
                             "trackdone" if passive else "clickrow", full=True)
                    if not first_ms_full:
                        # Qualified but short (30-90s), then skipped away.
                        rec = records[-1]
                        rec["ms_played"] = int(rng.integers(31_000, 90_000))
                        rec["reason_end"] = "fwdbtn"
                        rec["ts_local"] = t_start + pd.Timedelta(milliseconds=rec["ms_played"])
                        t = rec["ts_local"] + pd.Timedelta(seconds=5)
                    for k in range(int(rng.poisson(1.5 if converts else 0.3))):
                        t = play(t, artist, f"{artist} Song {k + 2}", pkey, shuffle, "trackdone", full=True)
                    if converts:
                        for _ in range(int(rng.integers(1, 5))):
                            scheduled[d + int(rng.integers(1, 31))].append(artist)
                    elif rng.random() < 0.2:
                        # Comes back, but only AFTER the 30-day window: must not count.
                        scheduled[d + int(rng.integers(31, 120))].append(artist)
                else:
                    artist = str(rng.choice(known, p=weights))
                    reason = rng.choice(["trackdone", "clickrow", "fwdbtn", "playbtn"],
                                        p=[0.55, 0.25, 0.15, 0.05])
                    t = play(t, artist, f"{artist} Hit {rng.integers(1, 30)}", pkey, shuffle, str(reason))
            # Occasional podcast episode: no artist, must be filtered out.
            if rng.random() < 0.15:
                records.append({**records[-1], "master_metadata_album_artist_name": None,
                                "master_metadata_track_name": None, "spotify_track_uri": None,
                                "episode_name": "Some Podcast Episode", "episode_show_name": "A Show",
                                "ts_local": t + pd.Timedelta(minutes=30), "ms_played": 1_500_000})

    df = pd.DataFrame(records)
    # Exact duplicate rows appear in real exports; add a few.
    df = pd.concat([df, df.sample(frac=0.005, random_state=seed)], ignore_index=True)

    # Local wall-clock -> UTC, respecting daylight saving time. Times that do
    # not exist (spring forward) are shifted; ambiguous ones (fall back) drop.
    local = pd.to_datetime(df["ts_local"]).dt.tz_localize(
        "America/New_York", ambiguous="NaT", nonexistent="shift_forward")
    df = df[local.notna()].copy()
    df["ts"] = local[local.notna()].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df = df.drop(columns=["ts_local"]).sort_values("ts")

    # Write one file per year, named like Spotify's real files.
    df["year"] = df["ts"].str[:4]
    for i, (year, part) in enumerate(df.groupby("year")):
        rows = part.drop(columns=["year"]).to_dict(orient="records")
        rows = [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
                for r in rows]
        path = out_dir / f"Streaming_History_Audio_{year}_{i}.json"
        path.write_text(json.dumps(rows))
    print(f"  generated {len(df):,} sample plays ({new_count:,} new artists) in {out_dir}")
    return out_dir


if __name__ == "__main__":
    import config
    generate(config.SAMPLE_RAW_DIR)
