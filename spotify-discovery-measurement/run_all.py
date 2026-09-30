"""
run_all.py
==========
Runs the whole project with one command.

    python run_all.py            # your real Spotify data in data/raw/
    python run_all.py --sample   # synthetic demo data (no personal data needed)

ORDER OF STEPS (each one reads what the previous step wrote)
    0. master    raw JSON  -> data/master/spotify_master.csv (src/build_master.py)
    1. load      master CSV -> raw_streams table        (src/load_raw.py)
    2. models    SQL files -> staging/intermediate/marts (sql/, src/build_models.py)
    3. checks    independent recomputation; stops on any failure (src/checks.py)
    4. model     predict which discoveries convert      (src/model_conversion.py)
    5. causal    passive vs active discovery            (src/causal_passive_vs_active.py)
    6. experiment power analysis / A/B test results     (src/experiment.py)
    7. dashboard outputs/dashboard.html                  (src/build_dashboard.py)
"""

import argparse
import shutil
import time

import config
from src import (build_dashboard, build_master, build_models, causal_passive_vs_active,
                 checks, experiment, load_raw, model_conversion, sample_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Spotify discovery measurement pipeline")
    parser.add_argument("--sample", action="store_true",
                        help="generate and use synthetic data instead of data/raw/")
    parser.add_argument("--rebuild-master", action="store_true",
                        help="rebuild data/master/spotify_master.csv from the JSON files")
    args = parser.parse_args()

    # Sample mode uses separate folders so it can never overwrite real results.
    if args.sample:
        raw_dir, db_path, out_dir = config.SAMPLE_RAW_DIR, config.SAMPLE_DB_PATH, config.SAMPLE_OUTPUT_DIR
        exp_dir = out_dir / "experiment"   # no schedule here: shows the planning view
    else:
        raw_dir, db_path, out_dir = config.RAW_DIR, config.DB_PATH, config.OUTPUT_DIR
        exp_dir = config.EXPERIMENT_DIR

    started = time.time()

    def step(title: str) -> None:
        print(f"\n[{time.time() - started:5.1f}s] {title}")

    if args.sample:
        step("0. Generating sample data")
        sample_data.generate(raw_dir)

    master = None
    if not args.sample:
        # Real data flows JSON -> master CSV -> database. The master file is
        # built once; delete it (or pass --rebuild-master) after a new export.
        if args.rebuild_master or not config.MASTER_PATH.exists():
            step("0. Building master data file from the JSON export")
            build_master.build(raw_dir)
        master = config.MASTER_PATH

    step("1. Loading listening history")
    load_info = load_raw.load(raw_dir, db_path, master_path=master)

    step("2. Building SQL models")
    build_models.build(db_path)

    step("3. Data-quality checks")
    passed = checks.run_checks(db_path)

    step("4. Conversion model")
    model_conversion.run(db_path, out_dir)

    step("5. Passive vs active (causal)")
    if load_info["has_reason_fields"]:
        causal_passive_vs_active.run(db_path, out_dir)
    else:
        # Basic export: no reason_start, so the comparison is impossible.
        # Writing an explicit status lets the dashboard explain why.
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "causal_results.json").write_text(
            '{"status": "insufficient_data", "message": "This analysis needs the extended '
            'streaming history, which records how each song started (reason_start). Request it '
            'from Spotify\'s privacy settings."}')
        print("  skipped: export has no reason_start field")

    step("6. Self-experiment")
    experiment.run(db_path, out_dir, exp_dir)

    step("7. Dashboard")
    page = build_dashboard.build(db_path, out_dir, is_sample=args.sample, n_checks=len(passed))
    print(f"  review flagged sleep/background artists in {out_dir / 'background_artists.csv'}")

    # GitHub Pages publishes the docs/ folder. Only real results go there.
    if not args.sample:
        config.DOCS_DIR.mkdir(exist_ok=True)
        shutil.copy(page, config.DOCS_DIR / "index.html")
        print(f"  copied to {config.DOCS_DIR / 'index.html'} for GitHub Pages")

    print(f"\nDone in {time.time() - started:.1f}s. Open {page} in your browser.")


if __name__ == "__main__":
    main()
