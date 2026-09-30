"""
src/causal_passive_vs_active.py
===============================
STEP 5: Do artists I meet PASSIVELY (autoplay / queue) come back less often
than artists I CHOOSE (clicked the song), once the situation is held equal?

WHY A SIMPLE COMPARISON IS NOT ENOUGH
  Passive discoveries happen in different situations: more evenings, more
  speakers and TVs, more shuffle. If those situations also change how likely
  you are to come back, a raw passive-vs-active difference mixes the effect
  of the discovery path with the effect of the situation (confounding).

METHOD: propensity score matching (primary) + inverse probability weighting
  1. Propensity model: predict P(passive | pre-exposure context) with
     logistic regression (time of day, weekend, device, shuffle, offline,
     year, and a within-period time trend). Only PRE-exposure features are used (see
     src/features.py for why engagement features would bias the answer).
  2. Matching: each passive discovery is compared with its closest active
     discoveries by propensity score (up to 5, with replacement, within a
     caliper), and only inside the same year x device x part-of-day x
     offline x weekend cell. Passive discoveries with no close active match are dropped
     rather than paired badly.
  3. Estimate: the ATT, the average effect for the passive discoveries, i.e.
     "how different would their return rate have been if I had chosen them?"
  4. Robustness: inverse probability weighting (IPW) estimates the same
     quantity a different way. If both methods agree, the result does not
     hinge on one method's quirks.
  5. Diagnostics: standardized mean differences (SMD) before and after
     matching. After matching, |SMD| < 0.1 on every covariate means the two
     groups look alike on everything we measured.
  6. Uncertainty: the WHOLE procedure (propensity model + matching) is
     re-run on bootstrap resamples, so the interval includes the uncertainty
     of the propensity model itself. (Bootstrapping a matching estimator is
     an approximation; the IPW interval is the better-behaved cross-check.)

HONEST LIMITATION (repeated on the dashboard)
  Matching only balances what is measured. Mood, what you were doing, or
  whether a friend recommended the song are unmeasured, so this is evidence,
  not proof. The randomized self-experiment (src/experiment.py) is the design
  that removes that limitation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import FEATURE_LABELS, load_events, pre_exposure_features, time_trend


def fit_propensity(X: pd.DataFrame, t: np.ndarray) -> np.ndarray:
    """P(passive | context). Probabilities are clipped away from 0 and 1 so the
    logit and the IPW weights stay finite."""
    ps_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    ps_model.fit(X, t)
    return np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)


def match_att(ps: np.ndarray, t: np.ndarray, y: np.ndarray, caliper_sd: float,
              strata: np.ndarray, k: int = config.PSM_NEIGHBORS) -> dict:
    """k-nearest-neighbor matching on the logit propensity score, within
    exact-match cells (strata).

    * Exact matching on year x device x part of day x offline x weekend: listening habits shift a
      lot between years (new phones, new app features, new routines) and
      between contexts, so a passive discovery on a phone on a 2024 evening
      is only compared with active discoveries on a phone on a 2024 evening.
    * Matching on the logit (not the raw probability) is the standard choice:
      it spreads out scores that bunch up near 0 or 1, and the 0.2 SD caliper
      rule is defined on this scale.
    * Each passive discovery is compared with the average of up to k closest
      active discoveries inside the caliper (with replacement, so a good
      control can serve several passive rows).
    * Passive rows with no active neighbor inside the caliper are dropped
      rather than matched badly; the share kept is reported.

    Returns the ATT plus per-row weights (treated rows: 1 if matched;
    control rows: how much each counts), used for the balance check.
    """
    logit = np.log(ps / (1 - ps))
    caliper = caliper_sd * logit.std()
    w_treated = np.zeros(len(t))
    w_control = np.zeros(len(t))

    for s_value in np.unique(strata):
        treated = np.where((t == 1) & (strata == s_value))[0]
        control = np.where((t == 0) & (strata == s_value))[0]
        if len(treated) == 0 or len(control) == 0:
            continue                       # nothing to compare within this cell
        kk = min(k, len(control))
        nn = NearestNeighbors(n_neighbors=kk).fit(logit[control].reshape(-1, 1))
        dist, pos = nn.kneighbors(logit[treated].reshape(-1, 1))
        inside = dist <= caliper                       # (n_treated, kk) booleans
        n_inside = inside.sum(axis=1)
        kept = n_inside > 0
        w_treated[treated[kept]] = 1.0
        # Each kept treated row spreads a total weight of 1 over its neighbors.
        share = np.where(inside, 1.0 / np.maximum(n_inside, 1)[:, None], 0.0)
        np.add.at(w_control, control[pos[kept]].ravel(), share[kept].ravel())

    n_treated = int((t == 1).sum())
    if w_treated.sum() == 0:
        return {"att": np.nan, "n_matched": 0, "n_treated": n_treated}
    treated_rate = float(np.average(y, weights=w_treated))
    control_rate = float(np.average(y, weights=w_control))
    return {
        "att": treated_rate - control_rate,
        "treated_rate": treated_rate,
        "control_rate": control_rate,
        "n_matched": int(w_treated.sum()),
        "n_treated": n_treated,
        "w_treated": w_treated,
        "w_control": w_control,
    }


def ipw_att(ps: np.ndarray, t: np.ndarray, y: np.ndarray) -> float:
    """ATT by weighting: treated rows weight 1, control rows weight ps/(1-ps).

    The weight makes the active group resemble the passive group: an active
    discovery in a very "passive-looking" situation counts for more.
    """
    w = ps[t == 0] / (1 - ps[t == 0])
    return float(y[t == 1].mean() - np.average(y[t == 0], weights=w))


def smd(X: pd.DataFrame, w_t: np.ndarray, w_c: np.ndarray, pooled_sd: pd.Series) -> pd.Series:
    """Weighted standardized mean difference between the two groups.

    The SAME pre-matching pooled SD is used before and after matching, so the
    two numbers are directly comparable. Before matching the weights are just
    0/1 group membership; after matching they are the matching weights.
    """
    mean_t = (X.mul(w_t, axis=0)).sum() / w_t.sum()
    mean_c = (X.mul(w_c, axis=0)).sum() / w_c.sum()
    return (mean_t - mean_c) / pooled_sd.replace(0, np.nan)


def run(db_path: Path = config.DB_PATH, output_dir: Path = config.OUTPUT_DIR) -> dict:
    df = load_events(db_path, eligible_only=True)
    df = df[df["discovery_path"].isin(["active", "passive"])].reset_index(drop=True)
    n_t = int((df["discovery_path"] == "passive").sum())
    n_c = int((df["discovery_path"] == "active").sum())

    if n_t < config.MIN_GROUP_SIZE or n_c < config.MIN_GROUP_SIZE:
        result = {"status": "insufficient_data", "n_passive": n_t, "n_active": n_c,
                  "message": ("This needs the extended streaming history (reason_start field) and "
                              f"at least {config.MIN_GROUP_SIZE} discoveries per path; "
                              f"found {n_t} passive and {n_c} active.")}
        _save(result, output_dir)
        print(f"  causal analysis skipped: {result['message']}")
        return result

    X = pre_exposure_features(df)
    X["months_since_start"] = time_trend(df)
    # Year dummies let the propensity model capture era shifts (e.g. a year
    # of heavy autoplay) that a straight-line trend cannot.
    years = pd.to_datetime(df["first_listen_date"]).dt.year
    X = X.join(pd.get_dummies(years, prefix="year", dtype=int))
    X = X.loc[:, X.std() > 0]                    # constant columns carry no information
    # Exact-match cells: same year, same device group, same part of the day,
    # same offline status, weekday vs weekend.
    # Propensity matching alone left late-night and speaker listening
    # imbalanced on the real data; matching exactly on the strongest
    # confounders fixes that, and the propensity score balances the rest
    # (weekend, shuffle, offline, within-year trend) inside each cell.
    daypart = np.select([X.get("daypart_morning", 0) == 1, X.get("daypart_afternoon", 0) == 1,
                         X.get("daypart_evening", 0) == 1], ["morning", "afternoon", "evening"], "night")
    # Offline plays are rare, so they are matched exactly too: a handful of
    # reused matches can otherwise swing their balance by chance. Weekend is
    # matched exactly because it stayed imbalanced when left to the score.
    strata = (years.astype(str) + "|" + df["platform_group"] + "|" + daypart
              + "|" + df["offline"].fillna(0).astype(int).astype(str)
              + "|" + X["is_weekend"].astype(str)).to_numpy()
    t = (df["discovery_path"] == "passive").to_numpy().astype(int)
    y = df["converted"].to_numpy().astype(float)

    # --- Point estimates on the full data --------------------------------
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    ps = fit_propensity(X, t)
    m = match_att(ps, t, y, config.PSM_CALIPER_SD, strata)
    ipw = ipw_att(ps, t, y)

    # --- Balance diagnostics ---------------------------------------------
    pooled_sd = np.sqrt((X[t == 1].var() + X[t == 0].var()) / 2)
    before = smd(X, (t == 1).astype(float), (t == 0).astype(float), pooled_sd)
    after = smd(X, m["w_treated"], m["w_control"], pooled_sd)
    # Year dummies are matched exactly, so their balance is shown as one
    # "year" line (the largest year SMD) instead of ten separate rows.
    year_cols = [c for c in X.columns if c.startswith("year_")]
    other_cols = [c for c in X.columns if c not in year_cols]
    balance = [{"covariate": FEATURE_LABELS.get(c, c),
                "smd_before": round(float(before[c]), 3),
                "smd_after": round(float(after[c]), 3)} for c in other_cols]
    if year_cols:
        worst = before[year_cols].abs().idxmax()
        balance.append({"covariate": "Calendar year (largest gap)",
                        "smd_before": round(float(before[worst]), 3),
                        "smd_after": round(float(after[year_cols].abs().max()), 3)})

    # --- Bootstrap: redo the entire procedure on resampled data ------------
    # Resampling passive and active rows separately keeps the group sizes
    # fixed, so every resample has enough of both.
    rng = np.random.default_rng(config.RANDOM_SEED)
    idx_t, idx_c = np.where(t == 1)[0], np.where(t == 0)[0]
    boot_psm, boot_ipw, boot_naive = [], [], []
    for _ in range(config.N_BOOTSTRAP):
        idx = np.concatenate([rng.choice(idx_t, len(idx_t)), rng.choice(idx_c, len(idx_c))])
        Xb, tb, yb = X.iloc[idx].reset_index(drop=True), t[idx], y[idx]
        psb = fit_propensity(Xb, tb)
        mb = match_att(psb, tb, yb, config.PSM_CALIPER_SD, strata[idx])
        if not np.isnan(mb["att"]):
            boot_psm.append(mb["att"])
        boot_ipw.append(ipw_att(psb, tb, yb))
        boot_naive.append(yb[tb == 1].mean() - yb[tb == 0].mean())

    def ci(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    # Propensity-score overlap for the dashboard: if the two histograms barely
    # overlap, matching is comparing unlike situations and the result is weak.
    bins = np.linspace(0, 1, 21)
    result = {
        "status": "ok",
        "n_passive": n_t, "n_active": n_c,
        "rates": {"passive": float(y[t == 1].mean()), "active": float(y[t == 0].mean())},
        "naive": {"estimate": naive, "ci95": ci(boot_naive)},
        "psm": {"estimate": m["att"], "ci95": ci(boot_psm),
                "n_matched": m["n_matched"], "n_treated": m["n_treated"],
                "share_matched": m["n_matched"] / m["n_treated"],
                "matched_passive_rate": m["treated_rate"],
                "matched_active_rate": m["control_rate"]},
        "ipw": {"estimate": ipw, "ci95": ci(boot_ipw)},
        "balance": balance,
        "max_abs_smd_after": float(after.abs().max()),
        "overlap": {"bins": bins.round(2).tolist(),
                    "passive": np.histogram(ps[t == 1], bins)[0].tolist(),
                    "active": np.histogram(ps[t == 0], bins)[0].tolist()},
    }
    _save(result, output_dir)
    print(f"  naive diff {naive:+.3f} | matched ATT {m['att']:+.3f} "
          f"(95% CI {result['psm']['ci95'][0]:+.3f} to {result['psm']['ci95'][1]:+.3f}) "
          f"| IPW {ipw:+.3f} | max |SMD| after {result['max_abs_smd_after']:.3f}")
    return result


def _save(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "causal_results.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    run()
