"""
src/experiment.py
=================
STEP 6: A real randomized A/B test on my own listening.

THE QUESTION
  Does listening with the treatment (default: Smart Shuffle on) change how
  many new artists I discover per day?

WHY THIS IS THE STRONGEST PART OF THE PROJECT
  The causal analysis in step 5 can only adjust for what is measured. A coin
  flip is independent of everything (mood, schedule, weather) by design, so a
  randomized difference has a causal interpretation without those
  assumptions. This is the same logic behind Spotify's own A/B tests.

DESIGN (decided BEFORE collecting data, which is the point)
  * Unit: one calendar day.
  * Paired randomization: days are grouped into consecutive pairs and a coin
    flip picks which day of the pair gets the treatment. Pairs guarantee
    equal arms and cancel slow drifts (a busy week affects both days).
  * Outcome: new artists discovered that day, computed from the streaming
    export itself, not self-reported, so it can't be nudged by memory.
  * Analysis: intention-to-treat (days are analyzed as assigned, even if you
    forgot one day) using a paired t-test plus an exact randomization test.

HOW TO RUN IT
  1. python -m src.experiment plan      -> writes experiment/assignment_schedule.csv
  2. Follow the schedule each day. Switch modes when you wake up.
  3. When it ends, request a new Spotify export (the quick "account data"
     export is enough), drop it in data/raw/ and run:  python run_all.py
"""

import argparse
import itertools
import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import config

SCHEDULE_FILE = "assignment_schedule.csv"


# ---------------------------------------------------------------------------
# Outcome data
# ---------------------------------------------------------------------------
def daily_discoveries(db_path: Path) -> pd.DataFrame:
    """One row per calendar day: new artists discovered and streams played.

    Days with no listening are included with zeros. Leaving them out would
    silently drop exactly the days the treatment might have affected.
    Days inside the burn-in window are flagged, because "new" artists there
    are mostly artists you already knew.
    """
    with sqlite3.connect(db_path) as con:
        streams = pd.read_sql(
            "SELECT listen_date, SUM(is_qualified) AS qualified_streams "
            "FROM stg_foreground_streams GROUP BY listen_date", con)   # sleep/background excluded
        firsts = pd.read_sql(
            "SELECT first_listen_date AS listen_date, COUNT(*) AS new_artists "
            "FROM int_artist_first_listen GROUP BY first_listen_date", con)

        span = con.execute("SELECT MIN(listen_date), MAX(listen_date) FROM stg_streams").fetchone()

    days = pd.DataFrame({"listen_date": pd.date_range(span[0], span[1], freq="D")})
    for frame in (streams, firsts):
        frame["listen_date"] = pd.to_datetime(frame["listen_date"])
    days = days.merge(streams, on="listen_date", how="left").merge(firsts, on="listen_date", how="left")
    days = days.fillna({"qualified_streams": 0, "new_artists": 0})
    burn_in_end = days["listen_date"].min() + pd.Timedelta(days=config.BURN_IN_DAYS)
    days["measurable"] = days["listen_date"] >= burn_in_end
    return days


# ---------------------------------------------------------------------------
# Planning: power analysis and an A/A test on historical data
# ---------------------------------------------------------------------------
def historical_pair_diffs(daily: pd.DataFrame, lookback_days: int = 180) -> np.ndarray:
    """Differences between the two days of consecutive, non-overlapping pairs
    in recent history. Their spread is the noise the experiment must beat."""
    recent = daily[daily["measurable"]].tail(lookback_days)["new_artists"].to_numpy()
    n = len(recent) // 2 * 2
    pairs = recent[:n].reshape(-1, 2)
    return pairs[:, 0] - pairs[:, 1]


def minimum_detectable_effect(sd_diff: float, n_pairs: int,
                              alpha: float = config.EXPERIMENT_ALPHA,
                              power: float = config.EXPERIMENT_POWER) -> float:
    """Smallest true difference (new artists per day) the design detects with
    the chosen power. Uses t quantiles, which are more honest than the normal
    approximation when there are only a few dozen pairs."""
    df = n_pairs - 1
    return float((stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)) * sd_diff / np.sqrt(n_pairs))


def aa_false_positive_rate(daily: pd.DataFrame, n_pairs: int, n_sims: int = 1000,
                           seed: int = config.RANDOM_SEED) -> float:
    """Run many fake experiments on history where NOTHING changed.

    With alpha = 0.05 about 5% should come out "significant". A much higher
    rate would mean the analysis is broken (e.g. strong day-to-day
    dependence), so this checks the method before trusting it.
    """
    series = daily[daily["measurable"]]["new_artists"].to_numpy()
    span = 2 * n_pairs
    if len(series) < span + 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_sims):
        start = rng.integers(0, len(series) - span)
        pairs = series[start:start + span].reshape(-1, 2)
        flip = rng.integers(0, 2, n_pairs)                    # coin flip per pair
        treat = np.where(flip == 1, pairs[:, 0], pairs[:, 1])
        ctrl = np.where(flip == 1, pairs[:, 1], pairs[:, 0])
        diffs = treat - ctrl
        if diffs.std() == 0:
            continue
        if stats.ttest_1samp(diffs, 0).pvalue < config.EXPERIMENT_ALPHA:
            hits += 1
    return hits / n_sims


def power_table(daily: pd.DataFrame) -> dict:
    diffs = historical_pair_diffs(daily)
    if len(diffs) < 10:
        return {"status": "insufficient_data"}
    sd = float(diffs.std(ddof=1))
    mean_daily = float(daily[daily["measurable"]].tail(180)["new_artists"].mean())
    rows = []
    for n in [7, 14, 21, 28, 42]:
        mde = minimum_detectable_effect(sd, n)
        rows.append({"n_pairs": n, "days": 2 * n, "mde_artists_per_day": round(mde, 2),
                     "mde_pct_of_mean": round(100 * mde / mean_daily, 1) if mean_daily else None})
    return {"status": "ok", "sd_pair_diff": sd, "mean_new_artists_per_day": mean_daily,
            "table": rows,
            "aa_false_positive_rate": aa_false_positive_rate(daily, config.EXPERIMENT_N_PAIRS)}


# ---------------------------------------------------------------------------
# The schedule (randomization)
# ---------------------------------------------------------------------------
def next_monday(today: date) -> date:
    return today + timedelta(days=(7 - today.weekday()) % 7 or 7)


def make_schedule(start: date, n_pairs: int, seed: int) -> pd.DataFrame:
    """Paired coin flips. Seeded, so the schedule is reproducible and can be
    committed to GitHub BEFORE the experiment starts (pre-registration)."""
    rng = np.random.default_rng(seed)
    rows = []
    for pair in range(n_pairs):
        first_is_treatment = rng.integers(0, 2) == 1
        for offset, is_treat in ((0, first_is_treatment), (1, not first_is_treatment)):
            day = start + timedelta(days=2 * pair + offset)
            rows.append({"date": day.isoformat(), "pair_id": pair + 1,
                         "arm": "treatment" if is_treat else "control",
                         "instruction": (config.EXPERIMENT_TREATMENT_LABEL if is_treat
                                         else config.EXPERIMENT_CONTROL_LABEL)})
    return pd.DataFrame(rows)


def plan(experiment_dir: Path = config.EXPERIMENT_DIR) -> Path:
    """Create the schedule once. Refuses to overwrite, because re-rolling the
    coin after seeing results would break the randomization."""
    path = experiment_dir / SCHEDULE_FILE
    if path.exists():
        print(f"Schedule already exists at {path}. Delete it only if the experiment has not started.")
        return path
    start = (date.fromisoformat(config.EXPERIMENT_START_DATE) if config.EXPERIMENT_START_DATE
             else next_monday(date.today()))
    schedule = make_schedule(start, config.EXPERIMENT_N_PAIRS, config.RANDOM_SEED + start.toordinal())
    experiment_dir.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(path, index=False)
    print(f"Wrote {len(schedule)} days ({start} to {schedule['date'].iloc[-1]}) to {path}")
    return path


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def randomization_p_value(diffs: np.ndarray, max_exact_pairs: int = 16,
                          n_random: int = 20000, seed: int = config.RANDOM_SEED) -> float:
    """Exact test that matches the design: under "no effect", each pair's
    treatment/control labels could equally have been swapped, which flips the
    sign of its difference. The p-value is the share of all sign patterns
    with a mean at least as extreme as the one observed."""
    observed = abs(diffs.mean())
    n = len(diffs)
    if n <= max_exact_pairs:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_random, n))
    null_means = np.abs((signs * diffs).mean(axis=1))
    return float((null_means >= observed - 1e-12).mean())


def analyze(daily: pd.DataFrame, schedule: pd.DataFrame) -> dict:
    """Paired, intention-to-treat analysis of a finished experiment."""
    sched = schedule.assign(date=pd.to_datetime(schedule["date"]))
    data = sched.merge(daily, left_on="date", right_on="listen_date", how="left")
    wide = data.pivot(index="pair_id", columns="arm", values="new_artists")
    diffs = (wide["treatment"] - wide["control"]).to_numpy(dtype=float)
    n = len(diffs)
    mean_diff = float(diffs.mean())
    se = float(diffs.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    t_crit = stats.t.ppf(0.975, n - 1)
    ctrl_mean = float(wide["control"].mean())
    return {
        "n_pairs": n,
        "treatment_mean": float(wide["treatment"].mean()),
        "control_mean": ctrl_mean,
        "mean_diff": mean_diff,
        "ci95": [mean_diff - t_crit * se, mean_diff + t_crit * se],
        "lift_pct": 100 * mean_diff / ctrl_mean if ctrl_mean else None,
        "p_value_t": float(stats.ttest_1samp(diffs, 0).pvalue) if diffs.std() > 0 else 1.0,
        "p_value_randomization": randomization_p_value(diffs),
        "pairs": [{"pair_id": int(i), "treatment": float(r["treatment"]), "control": float(r["control"])}
                  for i, r in wide.iterrows()],
    }


def run(db_path: Path = config.DB_PATH, output_dir: Path = config.OUTPUT_DIR,
        experiment_dir: Path = config.EXPERIMENT_DIR) -> dict:
    daily = daily_discoveries(db_path)
    result = {"treatment_label": config.EXPERIMENT_TREATMENT_LABEL,
              "control_label": config.EXPERIMENT_CONTROL_LABEL,
              "n_pairs_planned": config.EXPERIMENT_N_PAIRS,
              "power": power_table(daily)}

    path = experiment_dir / SCHEDULE_FILE
    if not path.exists():
        result["status"] = "not_planned"
    else:
        schedule = pd.read_csv(path)
        data_end = daily["listen_date"].max()
        dates = pd.to_datetime(schedule["date"])
        result["schedule_start"] = dates.min().date().isoformat()
        result["schedule_end"] = dates.max().date().isoformat()
        covered = int((dates <= data_end).sum())
        if covered == 0:
            result["status"] = "not_started"
        elif covered < len(schedule):
            result["status"] = "in_progress"
            result["days_covered"] = covered
            result["days_total"] = int(len(schedule))
        else:
            result["status"] = "complete"
            result["analysis"] = analyze(daily, schedule)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "experiment_results.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"  experiment status: {result['status']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-experiment tools")
    parser.add_argument("command", choices=["plan", "analyze"],
                        help="plan = create the randomized schedule; analyze = evaluate it")
    args = parser.parse_args()
    if args.command == "plan":
        plan()
    else:
        run()
