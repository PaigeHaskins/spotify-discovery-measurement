# Did discovery stick? Measuring music discovery in 10 years of my Spotify history

A measurement project in the spirit of Spotify's **Discovery Mode**: define what a successful
discovery is, try to predict it from the first listen, test whether *how* a song reaches me
changes it, and design a randomized experiment to check.

**Live dashboard:** ["View Interactive Dashboard"](https://paigehaskins.github.io/spotify-discovery-measurement/) 

## Results (Oct 2016 to Sep 2026)

**First, sleep and background listening is set aside.** I fall asleep to brown noise and let background
playlists run for hours, so 1,972 hours (25% of all listening) are excluded before any analysis: 554
sleep/noise artists found by album and track titles, plus 270 sessions where autoplay ran for hours with
almost no skips or clicks (821 artists heard almost only in those sessions are removed entirely). Left in,
they would inflate results: set-aside artists "came back" 40% of the time because the same sleep sounds
play every night. 220,457 of 256,881 plays remain.

| Question | Answer |
|---|---|
| How often does a new artist stick? | **24%** of 2,914 new artists got a return visit on a later day within 30 days. |
| Can the first listen predict it? | **Modestly.** On future discoveries (time-based split) the model's AUC is 0.60 (95% CI 0.55 to 0.66), and its top 10% of picks return at 1.9x the average rate. The strongest signal is how many times I played the artist on the discovery day. |
| Do autoplay discoveries stick less than ones I click? | **Yes, somewhat.** Compared with clicked discoveries in the same year, device, part of day, and kind of day, autoplay discoveries returned 5.3 points less often (95% CI -11.9 to -0.8). Inverse-probability weighting agrees (-8.6 points). All covariates balanced after matching. |
| Randomized experiment | Designed (paired daily coin flips, power analysis, A/A test); ready to run. |

Removing background listening changed the conclusions: with sleep sounds included, the model looked
no better than a coin flip and the autoplay gap disappeared. Defining the population correctly was the
most important analytical decision in the project.

## What this project demonstrates

- **Metric design:** stream (30s+), discovery, and 30-day return definitions, with fixes for left-censoring (burn-in) and right-censoring (incomplete windows), and a behavioral rule that separates attentive listening from sleep/background playback.
- **SQL modeling:** staging, intermediate, and mart layers (dbt-style), including sessionization with window functions, with 17 automated data-quality checks that recompute key numbers a second, independent way.
- **ML model evaluation:** time-based split, baseline, bootstrap confidence intervals, calibration, top-10% lift, challenger model, and drift diagnostics.
- **Causal inference:** propensity score matching within exact-match cells, IPW cross-check, balance diagnostics, and bootstrapped uncertainty.
- **Experimentation:** paired randomization, power analysis, A/A test, and an exact randomization test.
- **Validation:** 16 automated tests, including synthetic data with planted effects that the analysis must recover.

## Project structure

```
config.py                         every definition and threshold in one place
run_all.py                        runs the whole pipeline
sql/
  01_stg_streams.sql              one clean row per play
  02_int_listening_sessions.sql   sessions; flags long untouched (background) sessions
  03_int_background_artists.sql   sleep/noise and background-only artists, with reasons
  04_stg_foreground_streams.sql   attentive listening only (all analysis reads this)
  05_int_artist_first_listen.sql  each artist's first 30s+ stream
  06_mart_discovery_events.sql    discovery events, return label, eligibility
  07_mart_monthly_kpis.sql        monthly program-health KPIs
src/
  build_master.py                 JSON export -> data/master/spotify_master.csv
  load_raw.py                     master CSV -> SQLite (drops personal identifiers)
  build_models.py                 runs the SQL models in order
  checks.py                       data-quality checks (pipeline stops on failure)
  features.py                     shared feature engineering
  model_conversion.py             predictive model + evaluation
  causal_passive_vs_active.py     matching, IPW, balance diagnostics
  experiment.py                   schedule, power analysis, A/A test, analysis
  build_dashboard.py              self-contained HTML dashboard
  sample_data.py                  synthetic export with planted effects
tests/test_pipeline.py            automated tests
docs/index.html                   the published dashboard (GitHub Pages)
```

## Run it

```bash
pip install -r requirements.txt

# Option A: your own data
# 1. Request "Extended streaming history" in Spotify's privacy settings (takes up to 30 days).
# 2. Unzip the download into data/raw/
python run_all.py            # builds the master file, the database, and outputs/dashboard.html

# Option B: no data needed
python run_all.py --sample   # synthetic listening history, clearly labeled as sample data

# Tests
python tests/test_pipeline.py
```

After a new Spotify export, run `python run_all.py --rebuild-master`.

## Review the sleep/background rules

Each run writes `outputs/background_artists.csv`: every set-aside artist, their hours, and why they were
flagged. If a real artist is on the list, add the name to `BACKGROUND_NEVER` in `config.py`; to set one
aside by hand, add it to `BACKGROUND_ALWAYS`. The thresholds are also in `config.py`.

## Run the experiment

```bash
python -m src.experiment plan   # writes experiment/assignment_schedule.csv
```

Commit the schedule to GitHub **before** starting (it proves the coin flips were fixed in advance),
follow it daily, then add a fresh export and rerun `python run_all.py`.

## Publish the dashboard

`run_all.py` copies the dashboard to `docs/index.html`. In the GitHub repo, open Settings, then Pages,
and serve from the `main` branch, `/docs` folder. Note that this makes your listening results public;
set `SHOW_ARTIST_NAMES = False` in `config.py` to hide artist names.

## Privacy

Raw exports, the master CSV, and the database are excluded by `.gitignore`. IP addresses are dropped
when the master file is built.

## Limitations

One listener, so nothing here generalizes to Spotify's users. Artists are matched by name (the export
has no artist ID). The export records how a song started, not which playlist or algorithm served it, so
this measures discovery behavior, not Spotify's recommendation systems. Matching adjusts only for
measured context.

## Porting to BigQuery and dbt

The SQL uses SQLite. To move it to dbt on BigQuery, replace `strftime()`/`DATE(x, '+N days')` with
`FORMAT_DATE`/`DATE_ADD`, and the `{{PLACEHOLDERS}}` with dbt `var()` calls. The logic is unchanged.

