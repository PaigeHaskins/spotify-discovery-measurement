"""
src/model_conversion.py
=======================
STEP 4: Can the first encounter predict whether a new artist sticks?

WHAT IT DOES
  Trains two classifiers on eligible discovery events and evaluates them the
  way a model-evaluation lead would: against a baseline, on held-out FUTURE
  data, with uncertainty, calibration, and a business-style lift metric.

KEY DESIGN DECISIONS (and why)
  1. Time-based split, not a random split. The model is trained on older
     discoveries and tested on the most recent ones. A production model always
     predicts the future from the past; a random split would leak future
     listening habits into training and overstate performance.
  2. A pre-declared primary model. Logistic regression is the headline model
     because it is interpretable. Gradient boosting is reported as a
     challenger. Picking whichever scores best on the test set would be a
     quiet form of overfitting to the test set.
  3. A baseline. Every metric is shown next to what "always predict the
     average rate" would score, so improvement is visible, not assumed.
  4. Several metrics, because each answers a different question:
       ROC AUC      - does it rank converters above non-converters?
       Avg precision- same, but focused on the (often rarer) converters
       Brier score  - are the predicted probabilities accurate?
       Top-10% lift - if we promoted only the top 10% of predictions, how much
                      better would their conversion rate be than average?
                      (the campaign-optimization framing Discovery Mode uses)
  5. Bootstrap confidence interval on test AUC, because a test set of a few
     hundred rows gives a noisy AUC and the interval says how noisy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import FEATURE_LABELS, load_events, model_matrix


def time_split(df: pd.DataFrame, test_share: float):
    """Oldest (1 - test_share) of rows -> train, newest -> test.

    Rows are already sorted by first-listen time (see load_events). The split
    point is moved forward so both sides never share a calendar day, which
    prevents same-day information from leaking across the boundary.
    """
    cut = int(len(df) * (1 - test_share))
    cut_date = df.loc[cut, "first_listen_date"]
    train = df[df["first_listen_date"] < cut_date]
    test = df[df["first_listen_date"] >= cut_date]
    return train, test


def top_fraction_lift(y: np.ndarray, p: np.ndarray, fraction: float = 0.10) -> dict:
    """Conversion rate among the top `fraction` of predictions vs overall."""
    k = max(1, int(round(len(p) * fraction)))
    top = np.argsort(-p)[:k]
    rate_top = float(y[top].mean())
    base = float(y.mean())
    return {"top_rate": rate_top, "base_rate": base,
            "lift": rate_top / base if base > 0 else None, "n_top": int(k)}


def bootstrap_auc(y: np.ndarray, p: np.ndarray, n: int, seed: int) -> list:
    """95% percentile interval for AUC by resampling test rows with replacement."""
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():   # a resample with one class has no AUC
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def evaluate(name: str, y: np.ndarray, p: np.ndarray, seed: int) -> dict:
    """All evaluation metrics for one model's test-set predictions."""
    fpr, tpr, _ = roc_curve(y, p)
    # Thin the ROC curve to at most ~150 points to keep the dashboard light.
    step = max(1, len(fpr) // 150)
    n_bins = int(min(10, max(3, len(y) // 40)))
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y, p)),
        "roc_auc_ci95": bootstrap_auc(y, p, config.N_BOOTSTRAP, seed),
        "avg_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "top10": top_fraction_lift(y, p, 0.10),
        "roc_curve": {"fpr": fpr[::step].round(4).tolist() + [1.0],
                      "tpr": tpr[::step].round(4).tolist() + [1.0]},
        "calibration": {"predicted": mean_pred.round(4).tolist(),
                        "observed": frac_pos.round(4).tolist()},
    }


def run(db_path: Path = config.DB_PATH, output_dir: Path = config.OUTPUT_DIR) -> dict:
    df = load_events(db_path, eligible_only=True)
    result = {"status": "ok", "n_eligible": int(len(df))}

    # Guard: too little data gives unstable, misleading numbers.
    if len(df) < 4 * config.MIN_GROUP_SIZE or df["converted"].nunique() < 2:
        result.update(status="insufficient_data",
                      message=f"Need at least {4 * config.MIN_GROUP_SIZE} eligible discoveries "
                              f"with both outcomes; found {len(df)}.")
        _save(result, output_dir)
        print(f"  model skipped: {result['message']}")
        return result

    X = model_matrix(df)
    y = df["converted"].to_numpy()
    train, test = time_split(df, config.TEST_SHARE)
    X_tr, X_te = X.loc[train.index], X.loc[test.index]
    y_tr, y_te = y[train.index], y[test.index]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or len(test) < config.MIN_GROUP_SIZE:
        result.update(status="insufficient_data",
                      message="The train or test period has only one outcome class or too few rows.")
        _save(result, output_dir)
        print(f"  model skipped: {result['message']}")
        return result

    # Primary model: scaled logistic regression. Scaling puts every
    # coefficient on a "per 1 standard deviation" footing so they compare.
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=1.0))
    logit.fit(X_tr, y_tr)
    p_logit = logit.predict_proba(X_te)[:, 1]

    # Challenger: shallow gradient boosting, which can find interactions
    # (e.g. "skipped AND on shuffle") that a linear model misses. Kept small
    # (depth 3, 20+ rows per leaf) because the data set is small.
    gbm = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                         min_samples_leaf=20, random_state=config.RANDOM_SEED)
    gbm.fit(X_tr, y_tr)
    p_gbm = gbm.predict_proba(X_te)[:, 1]

    # Baseline: predict the training conversion rate for everyone.
    base_rate = float(y_tr.mean())
    p_base = np.full(len(y_te), base_rate)

    result.update({
        "split": {"train_n": int(len(train)), "test_n": int(len(test)),
                  "train_period": [train["first_listen_date"].min(), train["first_listen_date"].max()],
                  "test_period": [test["first_listen_date"].min(), test["first_listen_date"].max()],
                  "train_conversion_rate": base_rate,
                  "test_conversion_rate": float(y_te.mean())},
        "primary": evaluate("Logistic regression", y_te, p_logit, config.RANDOM_SEED),
        "challenger": evaluate("Gradient boosting", y_te, p_gbm, config.RANDOM_SEED + 1),
        "baseline": {"roc_auc": 0.5,
                     "avg_precision": float(y_te.mean()),
                     "brier": float(brier_score_loss(y_te, p_base))},
    })

    # Drift check 1: the same model scored on a RANDOM 75/25 split. If it does
    # much better than on the time split, the patterns change over time and a
    # random split would have overstated real-world performance.
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y, test_size=config.TEST_SHARE,
                                                  stratify=y, random_state=config.RANDOM_SEED)
    rand_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    rand_model.fit(Xr_tr, yr_tr)
    result["random_split_auc"] = float(roc_auc_score(yr_te, rand_model.predict_proba(Xr_te)[:, 1]))

    # Drift check 2: signal stability. Each feature's own AUC in the training
    # years vs the test years. A signal near 0.5 carries no information; one
    # that crosses 0.5 has flipped direction. This is a "model health" view.
    stability = []
    for col in X.columns:
        if X_tr[col].std() > 0 and X_te[col].std() > 0:
            a_tr = roc_auc_score(y_tr, X_tr[col])
            a_te = roc_auc_score(y_te, X_te[col])
            stability.append({"feature": FEATURE_LABELS.get(col, col),
                              "auc_train": round(float(a_tr), 3), "auc_test": round(float(a_te), 3),
                              "strength": abs(a_tr - 0.5)})
    stability.sort(key=lambda r: -r["strength"])
    result["signal_stability"] = [{k: v for k, v in r.items() if k != "strength"} for r in stability[:8]]

    # Interpretability: odds ratio per +1 SD of each feature (primary model).
    # OR > 1 = more likely to come back; OR < 1 = less likely.
    coefs = logit.named_steps["logisticregression"].coef_[0]
    effects = pd.DataFrame({"feature": X.columns, "odds_ratio": np.exp(coefs)})
    effects = effects[X_tr.std().to_numpy() > 0]          # drop constant columns
    effects["abs_log"] = np.log(effects["odds_ratio"]).abs()
    effects = effects.sort_values("abs_log", ascending=False).head(8)
    result["feature_effects"] = [
        {"feature": FEATURE_LABELS.get(f, f), "odds_ratio": round(float(o), 3)}
        for f, o in zip(effects["feature"], effects["odds_ratio"])]

    # Challenger importance measured the model-agnostic way: how much test AUC
    # drops when one feature is shuffled.
    perm = permutation_importance(gbm, X_te, y_te, scoring="roc_auc", n_repeats=10,
                                  random_state=config.RANDOM_SEED)
    order = np.argsort(-perm.importances_mean)[:8]
    result["challenger_importance"] = [
        {"feature": FEATURE_LABELS.get(X.columns[i], X.columns[i]),
         "auc_drop": round(float(perm.importances_mean[i]), 4)} for i in order]

    _save(result, output_dir)
    pr = result["primary"]
    print(f"  primary AUC {pr['roc_auc']:.3f} (95% CI {pr['roc_auc_ci95'][0]:.3f}-"
          f"{pr['roc_auc_ci95'][1]:.3f}); top-10% lift {pr['top10']['lift']:.2f}x")
    return result


def _save(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_results.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    run()
