"""
tests/test_pipeline.py
======================
Automated tests. Run with either:
    python -m pytest tests/        (if pytest is installed)
    python tests/test_pipeline.py  (no extra install needed)

WHAT THESE TESTS PROVE
  The pipeline's data-quality checks (src/checks.py) confirm the tables are
  internally consistent. These tests go further: they build synthetic data
  where the TRUE answers are known, then confirm the analysis finds them, and
  they exercise edge cases (basic export format, duplicate files, the
  experiment math) that the sample data alone would not cover.
"""

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # allow `python tests/...`

import config  # noqa: E402
from src import (build_dashboard, build_models, causal_passive_vs_active, checks,  # noqa: E402
                 experiment, load_raw, model_conversion, sample_data)

# Build the synthetic data set once and share it across tests (it takes a few seconds).
_TMP = Path(tempfile.mkdtemp(prefix="spotify_test_"))
_RAW, _DB, _OUT = _TMP / "raw", _TMP / "test.db", _TMP / "out"
_built = False


def _pipeline():
    global _built
    if not _built:
        config.N_BOOTSTRAP = 150            # fewer resamples keeps tests fast
        sample_data.generate(_RAW, seed=3)
        load_raw.load(_RAW, _DB)
        build_models.build(_DB)
        _built = True
    return _DB


# --- Loading ----------------------------------------------------------------
def test_loader_drops_pii_podcasts_and_duplicates():
    db = _pipeline()
    raw_json = []
    for f in sorted(_RAW.glob("*.json")):
        raw_json.extend(json.loads(f.read_text()))
    df = pd.DataFrame(raw_json)
    music = df[df["master_metadata_album_artist_name"].notna()]
    expected = len(music.drop_duplicates(subset=["ts", "master_metadata_album_artist_name",
                                                 "master_metadata_track_name", "ms_played"]))
    with sqlite3.connect(db) as con:
        cols = [r[1] for r in con.execute("PRAGMA table_info(raw_streams)")]
        n = con.execute("SELECT COUNT(*) FROM raw_streams").fetchone()[0]
    assert "ip_addr" not in cols and "conn_country" not in cols
    assert n == expected, f"expected {expected} music plays after dedupe, got {n}"


def test_local_time_conversion_handles_daylight_saving():
    # 2025-07-01 16:00 UTC is 12:00 in Boston (EDT, UTC-4);
    # 2025-01-15 17:00 UTC is 12:00 in Boston (EST, UTC-5).
    df = pd.DataFrame({"ts_utc": pd.to_datetime(["2025-07-01T16:00:00Z", "2025-01-15T17:00:00Z"], utc=True),
                       "artist_name": ["A", "B"], "track_name": ["x", "y"], "album_name": None,
                       "track_uri": None, "ms_played": [60000, 60000], "platform": None,
                       "reason_start": None, "reason_end": None, "shuffle": None, "skipped": None,
                       "offline": None, "incognito_mode": None, "source_format": "extended"})
    out = load_raw.clean(df, "America/New_York")
    assert out["ts_local"].str[11:16].tolist() == ["12:00", "12:00"]


def test_basic_export_format_and_combination():
    """Account-data files load, and overlap with extended history is not double counted."""
    tmp = Path(tempfile.mkdtemp())
    ext = [{"ts": "2025-03-01T12:00:00Z", "ms_played": 200000, "master_metadata_album_artist_name": "A",
            "master_metadata_track_name": "a1", "reason_start": "clickrow", "platform": "windows"}]
    basic = [{"endTime": "2025-03-01 12:00", "artistName": "A", "trackName": "a1", "msPlayed": 200000},  # overlap
             {"endTime": "2025-03-05 09:30", "artistName": "B", "trackName": "b1", "msPlayed": 45000}]    # new
    (tmp / "Streaming_History_Audio_2025_0.json").write_text(json.dumps(ext))
    (tmp / "StreamingHistory_music_0.json").write_text(json.dumps(basic))
    (tmp / "StreamingHistory_podcast_0.json").write_text(json.dumps([{"x": 1}]))  # must be ignored
    info = load_raw.load(tmp, tmp / "t.db")
    assert info["n_plays"] == 2
    assert info["formats"] == ["basic", "extended"]
    shutil.rmtree(tmp)


# --- Background listening ---------------------------------------------------------
def _play(ts, artist, album, reason_start="trackdone", reason_end="trackdone", ms=180000):
    return {"ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "ms_played": ms, "platform": "ios",
            "master_metadata_album_artist_name": artist, "master_metadata_track_name": f"{artist} song",
            "master_metadata_album_album_name": album, "reason_start": reason_start,
            "reason_end": reason_end, "shuffle": False, "skipped": False, "offline": False}


def _background_fixture(tmp: Path) -> Path:
    """Three kinds of listening with known answers:
       * Real Band: daytime, lots of skips and clicks        -> keep
       * Night Trio: 2-hour untouched autoplay from 11pm UTC -> background session artist
       * Noise Co: clicked in the daytime, but "Brown Noise"  -> sleep/noise content
    """
    rows, t = [], pd.Timestamp("2025-06-01 14:00", tz="UTC")
    for i in range(30):                                   # awake, interactive
        rows.append(_play(t, "Real Band", "Real Album", "clickrow" if i % 3 else "fwdbtn",
                          "fwdbtn" if i % 2 else "trackdone"))
        t += pd.Timedelta(minutes=3, seconds=10)
    rows.append(_play(t + pd.Timedelta(minutes=5), "Noise Co", "Deep Brown Noise for Sleep", "clickrow"))
    t = pd.Timestamp("2025-06-02 03:00", tz="UTC")         # 11pm in Boston
    for _ in range(40):                                    # 2 hours, nobody touches it
        rows.append(_play(t, "Night Trio", "Soft Evening"))
        t += pd.Timedelta(minutes=3, seconds=5)
    (tmp / "Streaming_History_Audio_2025_0.json").write_text(json.dumps(rows))
    return tmp


def test_background_rules_flag_the_right_artists():
    tmp = _background_fixture(Path(tempfile.mkdtemp()))
    load_raw.load(tmp, tmp / "bg.db")
    build_models.build(tmp / "bg.db")
    with sqlite3.connect(tmp / "bg.db") as con:
        flagged = dict(con.execute("SELECT artist_name, background_reason FROM int_background_artists"))
        fg_artists = {r[0] for r in con.execute("SELECT DISTINCT artist_name FROM stg_foreground_streams")}
    assert flagged == {"Night Trio": "background_sessions", "Noise Co": "sleep_or_noise_content"}, flagged
    assert fg_artists == {"Real Band"}
    shutil.rmtree(tmp)


def test_background_never_override_keeps_an_artist():
    tmp = _background_fixture(Path(tempfile.mkdtemp()))
    saved = config.BACKGROUND_NEVER
    config.BACKGROUND_NEVER = ("Night Trio",)
    try:
        load_raw.load(tmp, tmp / "bg.db")
        build_models.build(tmp / "bg.db")
        with sqlite3.connect(tmp / "bg.db") as con:
            flagged = {r[0] for r in con.execute("SELECT artist_name FROM int_background_artists")}
    finally:
        config.BACKGROUND_NEVER = saved
    # Night Trio is no longer a background ARTIST (its overnight plays still
    # sit in a background session, which is the intended behavior).
    assert flagged == {"Noise Co"}
    shutil.rmtree(tmp)


# --- SQL models -------------------------------------------------------------
def test_data_quality_checks_pass():
    assert len(checks.run_checks(_pipeline())) >= 17


def test_short_skip_is_not_a_discovery():
    """The generator sometimes plays a new artist for <30s before the real first
    listen. The discovery must be the qualified play, never the skip."""
    with sqlite3.connect(_pipeline()) as con:
        bad = con.execute(f"SELECT COUNT(*) FROM int_artist_first_listen "
                          f"WHERE first_ms_played < {config.MIN_PLAY_MS}").fetchone()[0]
    assert bad == 0


def test_return_after_window_does_not_count():
    """Some non-converting artists come back on day 31-120. They must be 0."""
    with sqlite3.connect(_pipeline()) as con:
        ev = pd.read_sql("SELECT * FROM mart_discovery_events WHERE eligibility='eligible'", con)
        st = pd.read_sql("SELECT artist_name, listen_date FROM stg_streams WHERE is_qualified=1", con)
    later = ev.merge(st, on="artist_name")
    gap = (pd.to_datetime(later["listen_date"]) - pd.to_datetime(later["first_listen_date"])).dt.days
    late_only = set(later.loc[gap > config.RETENTION_WINDOW_DAYS, "artist_name"]) - \
        set(later.loc[gap.between(1, config.RETENTION_WINDOW_DAYS), "artist_name"])
    assert late_only, "test data should contain late-only returners"
    assert ev.loc[ev["artist_name"].isin(late_only), "converted"].eq(0).all()


# --- Analyses recover the planted truth ---------------------------------------
def test_model_beats_baseline():
    res = model_conversion.run(_pipeline(), _OUT)
    assert res["status"] == "ok"
    assert res["primary"]["roc_auc_ci95"][0] > 0.6
    assert res["primary"]["brier"] < res["baseline"]["brier"]
    # Time split: no test discovery is older than any training discovery.
    assert res["split"]["train_period"][1] < res["split"]["test_period"][0]


def test_causal_recovers_planted_negative_effect():
    res = causal_passive_vs_active.run(_pipeline(), _OUT)
    assert res["status"] == "ok"
    assert res["psm"]["ci95"][1] < 0, "matched effect should be clearly negative"
    assert res["ipw"]["estimate"] < 0
    # Planted confounders push the raw gap further from zero than the adjusted one.
    assert res["naive"]["estimate"] < res["psm"]["estimate"]
    assert res["max_abs_smd_after"] < 0.1


def test_causal_reports_insufficient_data_without_reason_fields():
    tmp = Path(tempfile.mkdtemp())
    rows = [{"endTime": f"2025-{m:02d}-{d:02d} 12:00", "artistName": f"Art{m}{d}", "trackName": "t",
             "msPlayed": 120000} for m in range(1, 13) for d in range(1, 28)]
    (tmp / "StreamingHistory_music_0.json").write_text(json.dumps(rows))
    load_raw.load(tmp, tmp / "b.db")
    build_models.build(tmp / "b.db")
    res = causal_passive_vs_active.run(tmp / "b.db", tmp)
    assert res["status"] == "insufficient_data"
    shutil.rmtree(tmp)


# --- Experiment math -------------------------------------------------------------
def test_schedule_is_balanced_and_reproducible():
    from datetime import date
    a = experiment.make_schedule(date(2026, 10, 5), 14, seed=1)
    b = experiment.make_schedule(date(2026, 10, 5), 14, seed=1)
    assert a.equals(b)
    assert (a.groupby("pair_id")["arm"].nunique() == 2).all()      # one of each per pair
    assert len(a) == 28 and a["date"].is_unique


def test_experiment_detects_a_planted_effect():
    from datetime import date
    rng = np.random.default_rng(0)
    sched = experiment.make_schedule(date(2026, 1, 5), 20, seed=2)
    days = pd.DataFrame({"listen_date": pd.to_datetime(sched["date"])})
    days["new_artists"] = rng.poisson(3, len(days)) + np.where(sched["arm"] == "treatment", 3, 0)
    days["measurable"] = True
    res = experiment.analyze(days, sched)
    assert res["p_value_randomization"] < 0.05
    assert res["ci95"][0] < 3 < res["ci95"][1]


def test_randomization_test_is_exact_on_a_tiny_case():
    # With diffs [1, 1, 1] only 2 of the 8 sign patterns are as extreme -> p = 0.25.
    assert abs(experiment.randomization_p_value(np.array([1.0, 1.0, 1.0])) - 0.25) < 1e-9


def test_aa_false_positive_rate_near_alpha():
    daily = experiment.daily_discoveries(_pipeline())
    fpr = experiment.aa_false_positive_rate(daily, 14, n_sims=400)
    assert 0.01 <= fpr <= 0.10


# --- Dashboard ---------------------------------------------------------------
def test_dashboard_builds_with_every_section():
    db = _pipeline()
    model_conversion.run(db, _OUT)
    causal_passive_vs_active.run(db, _OUT)
    experiment.run(db, _OUT, _TMP / "no_schedule")
    page = build_dashboard.build(db, _OUT, is_sample=True, n_checks=12).read_text()
    for heading in ["First, setting aside", "How much am I discovering", "Can the first listen predict",
                    "Does it matter how I find", "Testing it with a coin flip", "How this was measured"]:
        assert heading in page
    assert page.count("<svg") >= 6
    assert "{" + "{" not in page and "nan%" not in page.lower()


if __name__ == "__main__":
    # Minimal runner so the tests work without pytest installed.
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:   # report every failure, then exit non-zero
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    shutil.rmtree(_TMP, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
