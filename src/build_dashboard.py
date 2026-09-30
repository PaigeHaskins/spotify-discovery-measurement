"""
src/build_dashboard.py
======================
STEP 7: Turn the results into one self-contained HTML dashboard.

WHY A STATIC HTML FILE (instead of a server app like Streamlit)
  * It opens by double-clicking, with nothing to install or keep running.
  * GitHub Pages can host it for free, so a recruiter gets a live link.
  * Charts are drawn with matplotlib and embedded as SVG, so the file has no
    external dependencies and looks sharp at any zoom level.

HOW IT'S ORGANIZED
  The page reads like an analysis write-up, in the order a hiring manager
  would ask the questions:
    0. What was set aside as sleep / background listening?    (data scope)
    1. How much am I discovering, and how much of it sticks?  (KPIs)
    2. Can the first listen predict which artists stick?       (model eval)
    3. Does HOW I find an artist matter?                       (causal)
    4. Testing it with a coin flip                             (experiment)
    5. How this was measured                                   (methods)
  Every section's headline sentence is generated from the numbers, and it
  softens automatically when a result is not statistically clear.
"""

import html
import io
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # draw to files, no screen needed
import matplotlib.dates
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Visual system: Spotify-inspired dark theme.
#   Black backgrounds, white type, and Spotify green as the main accent.
#   Two colors carry meaning in every chart:
#     ACTIVE (you chose the song)   = green
#     PASSIVE (it just played next) = amber, which stays distinct from green
#                                     for color-blind readers too
# ---------------------------------------------------------------------------
BG = "#121212"        # page background (Spotify's app black)
INK = "#FFFFFF"       # primary text
MUTED = "#B3B3B3"     # secondary text (Spotify's grey)
GRID = "#2A2A2A"      # gridlines and borders
ACTIVE = "#1DB954"    # Spotify green
PASSIVE = "#F59B23"   # amber
NEUTRAL = "#6A6A6A"   # baselines and "before" values

plt.rcParams.update({
    "svg.fonttype": "none",       # keep text as text so the page's font is used
    "svg.hashsalt": "discovery",  # stable SVG ids -> clean git diffs
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": MUTED, "legend.labelcolor": MUTED,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,       # gridlines behind bars, never on top
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "none", "axes.facecolor": "none",
})

_chart_counter = 0


def svg(fig) -> str:
    """Convert a matplotlib figure to an inline <svg> string.

    Several SVGs live on one page, so every internal id is prefixed with a
    chart number; otherwise two charts could point at each other's clip paths.
    """
    global _chart_counter
    _chart_counter += 1
    prefix = f"c{_chart_counter}-"
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    text = buf.getvalue()
    text = text[text.index("<svg"):]                                  # drop XML header
    text = re.sub(r'id="([^"]+)"', lambda m: f'id="{prefix}{m.group(1)}"', text)
    text = re.sub(r'url\(#([^)]+)\)', lambda m: f'url(#{prefix}{m.group(1)})', text)
    text = re.sub(r'xlink:href="#([^"]+)"', lambda m: f'xlink:href="#{prefix}{m.group(1)}"', text)
    # Let CSS control size: remove fixed width/height, keep the viewBox.
    text = re.sub(r'<svg([^>]*?) width="[^"]+" height="[^"]+"', r'<svg\1', text, count=1)
    return text.replace("<svg", '<svg class="chart" role="img"', 1)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------
def pct(x, digits=0):
    return "n/a" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def pts(x):
    """A difference between two rates, in percentage points, with its sign."""
    return f"{100 * x:+.1f} pts"


def esc(text) -> str:
    return html.escape(str(text))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"status": "missing"}


def ci_excludes_zero(ci) -> bool:
    return ci is not None and (ci[0] > 0 or ci[1] < 0)


def notice(message: str) -> str:
    """Shown in place of a section whose analysis could not run."""
    return f'<p class="notice">{esc(message)}</p>'


# ---------------------------------------------------------------------------
# Section 0: Sleep and background listening (set aside)
# ---------------------------------------------------------------------------
def background_return_rate(con) -> tuple:
    """How often set-aside artists 'came back', using the same 30-day rule.

    Shown to justify removing them: a noise track replayed every night looks
    like a discovery that converted, which would inflate the headline metric.
    """
    q = pd.read_sql(
        "SELECT s.artist_name, s.listen_date FROM stg_streams s "
        "JOIN int_background_artists b USING (artist_name) WHERE s.is_qualified = 1", con)
    if q.empty:
        return None, 0
    q["listen_date"] = pd.to_datetime(q["listen_date"])
    first = q.groupby("artist_name")["listen_date"].min().rename("first")
    q = q.join(first, on="artist_name")
    gap = (q["listen_date"] - q["first"]).dt.days
    end = q["listen_date"].max()
    full_window = first[first <= end - pd.Timedelta(days=config.RETENTION_WINDOW_DAYS)]
    came_back = q[gap.between(1, config.RETENTION_WINDOW_DAYS)]["artist_name"].unique()
    return float(full_window.index.isin(came_back).mean()), int(len(full_window))


def section_background(con, monthly: pd.DataFrame) -> tuple:
    artists = pd.read_sql("SELECT * FROM int_background_artists ORDER BY hours DESC", con)
    n_sessions = con.execute(
        "SELECT COUNT(*) FROM int_listening_sessions WHERE is_background_session = 1").fetchone()[0]
    total_h = con.execute("SELECT SUM(ms_played) / 3600000.0 FROM stg_streams").fetchone()[0]
    bg_h = float(monthly["background_hours"].sum())
    share = bg_h / total_h if total_h else 0
    info = {"background_hours": bg_h, "background_share": share,
            "background_artists": int(len(artists)), "background_sessions": int(n_sessions)}
    if bg_h == 0:
        return notice("No sleep or background listening was detected."), info

    # Yearly attentive vs set-aside hours (stacked): years stay readable at
    # any history length, and the stack shows the share at a glance.
    y = monthly.assign(year=monthly["month"].str[:4]).groupby("year")[
        ["listening_hours", "background_hours"]].sum()
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    ax.bar(y.index, y["listening_hours"], color=ACTIVE, width=0.7, label="Attentive listening")
    ax.bar(y.index, y["background_hours"], bottom=y["listening_hours"], color=NEUTRAL,
           width=0.7, label="Set aside")
    ax.set_ylabel("Hours")
    ax.set_title("Hours per year", loc="left", fontsize=11)
    ax.tick_params(axis="x", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    chart = svg(fig)

    reasons = artists["background_reason"].value_counts()
    n_content = int(reasons.get("sleep_or_noise_content", 0))
    n_session = int(reasons.get("background_sessions", 0))
    n_manual = int(reasons.get("manual", 0))
    label = {"sleep_or_noise_content": "sleep / noise", "background_sessions": "played while untouched",
             "manual": "added by hand"}
    top_html = ""
    if config.SHOW_ARTIST_NAMES and len(artists):
        items = "".join(f"<li><span>{esc(r.artist_name)}</span><span class=\"num\">"
                        f"{r.hours:.0f} h, {label.get(r.background_reason, r.background_reason)}</span></li>"
                        for r in artists.head(8).itertuples())
        top_html = f"<h3>Most-played set-aside artists</h3><ol class=\"stuck\">{items}</ol>"

    rate, n_rate = background_return_rate(con)
    rate_html = (f" Left in, they would have skewed the results: set-aside artists "
                 f"\"came back\" {pct(rate)} of the time, because the same sleep sounds play night "
                 f"after night." if rate is not None and n_rate >= config.MIN_GROUP_SIZE else "")
    body = f"""
      <p class="lede">I fall asleep to brown noise and let background playlists run for hours, so
      before measuring discovery I set aside <strong>{bg_h:,.0f} hours</strong>
      ({pct(share)} of all listening).</p>
      <p>Two rules decide what counts as background. By content: {n_content:,} artists whose album or
      track titles are mostly sleep or noise phrases (brown noise, rain sounds, sleep jazz). By behavior:
      {n_sessions:,} sessions where autoplay ran for hours with almost no skips or clicks, which happens when
      no one is listening; every play in them is set aside, and {n_session:,} artists heard almost only
      in those sessions are removed entirely{f", plus {n_manual} added by hand" if n_manual else ""}.
      {rate_html}</p>
      <div class="charts single"><figure>{chart}<figcaption>Set-aside hours are excluded from every
      metric below.</figcaption></figure></div>
      {top_html}"""
    return body, info


# ---------------------------------------------------------------------------
# Section 1: KPIs
# ---------------------------------------------------------------------------
def section_kpis(monthly: pd.DataFrame, events: pd.DataFrame) -> tuple:
    eligible = events[events["eligibility"] == "eligible"]
    n_disc = int((events["eligibility"] != "burn_in").sum())
    conv = float(eligible["converted"].mean()) if len(eligible) else None

    # Only months where discovery is measurable and the month is complete.
    m = monthly[(monthly["discovery_measurable"] == 1) & (monthly["is_partial_month"] == 0)].copy()
    m["month_start"] = pd.to_datetime(m["month_start"])
    charts = ""
    if len(m) >= 2:
        # Real dates on the x-axis (not text labels) so years of history stay
        # readable: matplotlib places one tick per year automatically.
        years = matplotlib.dates.YearLocator()
        year_fmt = matplotlib.dates.DateFormatter("%Y")

        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.bar(m["month_start"], m["new_artists"], color=ACTIVE, width=25)
        ax.set_title("New artists per month", loc="left", fontsize=11)
        ax.xaxis.set_major_locator(years); ax.xaxis.set_major_formatter(year_fmt)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", visible=False)
        charts += f'<figure>{svg(fig)}<figcaption>First-ever 30-second streams of an artist, after the warm-up period.</figcaption></figure>'

        c = m[m["eligible_discoveries"] >= 5].set_index("month_start")
        if len(c) >= 2:
            # Monthly rates bounce around, so the trend is a 6-month rolling
            # rate built from summed counts (not an average of rates), which
            # weights busy months correctly.
            roll = (c["converted_discoveries"].rolling(6, min_periods=3).sum()
                    / c["eligible_discoveries"].rolling(6, min_periods=3).sum())
            fig, ax = plt.subplots(figsize=(4.8, 2.8))
            ax.scatter(c.index, c["conversion_rate"], s=10, color=NEUTRAL, label="Month")
            ax.plot(roll.index, roll, color=ACTIVE, lw=2.2, label="6-month trend")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            ax.xaxis.set_major_locator(years); ax.xaxis.set_major_formatter(year_fmt)
            ax.tick_params(axis="x", labelsize=8)
            ax.set_title(f"Came back within {config.RETENTION_WINDOW_DAYS} days", loc="left", fontsize=11)
            ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
            ax.grid(axis="x", visible=False)
            charts += (f'<figure>{svg(fig)}<figcaption>Share of each month\'s new artists I streamed again '
                       f'on a later day. Months with fewer than 5 eligible discoveries are hidden.</figcaption></figure>')

    stuck = ""
    if config.SHOW_ARTIST_NAMES and len(eligible):
        top = eligible.sort_values(["return_days_in_window", "streams_on_discovery_day"],
                                   ascending=False).head(6)
        items = "".join(f"<li><span>{esc(r.artist_name)}</span><span class=\"num\">"
                        f"{int(r.return_days_in_window)} return days</span></li>"
                        for r in top.itertuples())
        stuck = f'<h3>Artists that stuck hardest</h3><ol class="stuck">{items}</ol>'

    body = f"""
      <p class="lede">I met <strong>{n_disc:,}</strong> new artists. Of those with a full
      {config.RETENTION_WINDOW_DAYS}-day window to observe, <strong>{pct(conv)}</strong>
      earned a return visit on a later day. That return rate is the conversion metric the
      rest of this page tries to predict and explain.</p>
      <div class="charts">{charts or notice("Not enough complete months to chart yet.")}</div>
      {stuck}"""
    return body, {"n_discoveries": n_disc, "n_eligible": int(len(eligible)), "conversion_rate": conv}


# ---------------------------------------------------------------------------
# Section 2: Model
# ---------------------------------------------------------------------------
def section_model(res: dict) -> tuple:
    if res.get("status") != "ok":
        return notice(res.get("message", "The model has not been run.")), {}
    p, ch, base = res["primary"], res["challenger"], res["baseline"]
    lo, hi = p["roc_auc_ci95"]

    # ROC curve: how well each model ranks converters above non-converters.
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1, label="Coin flip (0.50)")
    ax.plot(ch["roc_curve"]["fpr"], ch["roc_curve"]["tpr"], color=PASSIVE, lw=1.6,
            label=f"Gradient boosting ({ch['roc_auc']:.2f})")
    ax.plot(p["roc_curve"]["fpr"], p["roc_curve"]["tpr"], color=ACTIVE, lw=2.2,
            label=f"Logistic regression ({p['roc_auc']:.2f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Ranking quality (ROC)", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    roc = svg(fig)

    # Calibration: when the model says 40%, do about 40% come back?
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, ls="--", lw=1, label="Perfect calibration")
    ax.plot(p["calibration"]["predicted"], p["calibration"]["observed"], color=ACTIVE,
            marker="o", lw=2, label="Logistic regression")
    ax.set_xlabel("Predicted chance of return"); ax.set_ylabel("Actual return rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Are the probabilities honest?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    cal = svg(fig)

    # Odds ratios: which signals matter, and in which direction.
    fe = pd.DataFrame(res["feature_effects"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    colors = [ACTIVE if o >= 1 else PASSIVE for o in fe["odds_ratio"]]
    ax.barh(fe["feature"], np.log(fe["odds_ratio"]), color=colors, height=0.6)
    ax.axvline(0, color=INK, lw=0.8)
    ticks = np.array([0.25, 0.5, 1, 2, 4])
    lim = max(abs(np.log(fe["odds_ratio"])).max() * 1.15, np.log(1.5))
    ticks = ticks[np.abs(np.log(ticks)) <= lim]
    ax.set_xticks(np.log(ticks), [f"{t:g}x" for t in ticks])
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Odds of returning, per 1 SD increase")
    ax.set_title("What predicts a return", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    odds = svg(fig)

    t10 = p["top10"]
    good = lo > 0.5
    headline = (f"Yes. On discoveries the model never saw, it ranks future returners above "
                f"non-returners {p['roc_auc']:.2f} of the time (95% CI {lo:.2f} to {hi:.2f}; "
                f"a coin flip scores 0.50)." if good else
                f"Not reliably. On future discoveries the model's AUC is {p['roc_auc']:.2f}, and its 95% "
                f"interval ({lo:.2f} to {hi:.2f}) includes 0.50, a coin flip. The signals that "
                f"predicted returns in earlier years weakened or reversed later, which is exactly "
                f"the kind of drift a model-health check exists to catch.")
    s = res["split"]
    drift_html = ""
    if res.get("signal_stability"):
        rows = "".join(
            f"<tr><td>{esc(r['feature'])}</td><td>{r['auc_train']:.2f}</td><td>{r['auc_test']:.2f}</td>"
            f"<td>{'flipped' if (r['auc_train'] - 0.5) * (r['auc_test'] - 0.5) < 0 else ('weaker' if abs(r['auc_test'] - 0.5) < abs(r['auc_train'] - 0.5) else 'held or stronger')}</td></tr>"
            for r in res["signal_stability"])
        drift_html = f"""
        <h3>Did the signals hold up over time?</h3>
        <p>Each signal's own AUC in the training years vs the test years (0.50 = no information;
        crossing 0.50 means the direction flipped). The same model scored
        <strong>{res['random_split_auc']:.2f}</strong> on a random split vs {p['roc_auc']:.2f} on the
        time split: the gap is how much a random split would have overstated real-world accuracy.</p>
        <div class="table-wrap"><table>
          <thead><tr><th>Signal</th><th>AUC, training years</th><th>AUC, test years</th><th>Change</th></tr></thead>
          <tbody>{rows}</tbody></table></div>"""
    body = f"""
      <p class="lede">{headline}</p>
      <p>If I promoted only the top 10% of new artists by predicted score, {pct(t10['top_rate'])}
      of them would come back, versus {pct(t10['base_rate'])} overall: a
      <strong>{t10['lift']:.1f}x lift</strong>. {"That is the campaign-targeting framing: the model's value is in who it puts at the top of the list." if t10['lift'] >= 1.2 else "In campaign terms, targeting by this model would do no better than picking artists at random."}</p>
      <div class="charts"><figure>{roc}</figure><figure>{cal}</figure></div>
      <div class="charts single"><figure>{odds}<figcaption>Green bars raise the odds of a
      return, amber bars lower them. Scale is logarithmic, so 2x and 0.5x are equally strong.</figcaption></figure></div>
      <details><summary>Evaluation details</summary>
        <p>Trained on {s['train_n']:,} discoveries ({esc(s['train_period'][0])} to {esc(s['train_period'][1])}),
        tested on the next {s['test_n']:,} ({esc(s['test_period'][0])} to {esc(s['test_period'][1])}).
        Splitting by time, not at random, mirrors how a production model predicts the future from the past.
        Logistic regression was declared the primary model before testing; gradient boosting is the challenger.</p>
        <div class="table-wrap"><table>
          <thead><tr><th>Model</th><th>ROC AUC</th><th>Avg precision</th><th>Brier score</th><th>Top-10% lift</th></tr></thead>
          <tbody>
            <tr><td>Logistic regression (primary)</td><td>{p['roc_auc']:.3f}</td><td>{p['avg_precision']:.3f}</td><td>{p['brier']:.3f}</td><td>{p['top10']['lift']:.2f}x</td></tr>
            <tr><td>Gradient boosting</td><td>{ch['roc_auc']:.3f}</td><td>{ch['avg_precision']:.3f}</td><td>{ch['brier']:.3f}</td><td>{ch['top10']['lift']:.2f}x</td></tr>
            <tr><td>Baseline: predict the average</td><td>0.500</td><td>{base['avg_precision']:.3f}</td><td>{base['brier']:.3f}</td><td>1.00x</td></tr>
          </tbody></table></div>
        <p>Lower Brier score is better. Average precision for the baseline equals the test return rate.</p>
        {drift_html}
      </details>"""
    return body, {"auc": p["roc_auc"], "auc_ci": p["roc_auc_ci95"], "top10_lift": t10["lift"]}


# ---------------------------------------------------------------------------
# Section 3: Causal
# ---------------------------------------------------------------------------
def section_causal(res: dict) -> tuple:
    if res.get("status") != "ok":
        return notice(res.get("message", "The causal analysis has not been run.")), {}
    psm = res["psm"]

    # Estimate plot: the naive gap vs the two adjusted estimates.
    rows = [("Raw difference", res["naive"], NEUTRAL),
            ("Weighted (IPW)", res["ipw"], PASSIVE),
            ("Matched (primary)", psm, ACTIVE)]
    fig, ax = plt.subplots(figsize=(4.8, 2.6))
    for i, (label, est, color) in enumerate(rows):
        ax.plot(np.array(est["ci95"]) * 100, [i, i], color=color, lw=3, solid_capstyle="round")
        ax.plot(est["estimate"] * 100, i, "o", color=color, ms=9, mec=BG, mew=1.5)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("Passive minus active return rate (percentage points)")
    ax.set_title("Effect of discovering an artist passively", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    est_chart = svg(fig)

    # Balance ("love plot"): each dot should move inside the +/-0.1 band.
    bal = pd.DataFrame(res["balance"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.axvspan(-0.1, 0.1, color=GRID, alpha=0.7, lw=0)
    y = np.arange(len(bal))
    ax.plot(bal["smd_before"], y, "o", color=NEUTRAL, ms=6, label="Before matching")
    ax.plot(bal["smd_after"], y, "o", color=ACTIVE, ms=6, label="After matching")
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y, bal["covariate"])
    ax.set_xlabel("Standardized difference, passive vs active")
    ax.set_title("Were like situations compared?", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, -0.18), ncol=2)
    ax.grid(axis="y", visible=False)
    bal_chart = svg(fig)

    # Overlap of propensity scores between the two groups.
    ov = res["overlap"]
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    # Outlines instead of filled bars, so both groups stay visible where they overlap.
    edges = np.array(ov["bins"])
    ax.stairs(ov["active"], edges, color=ACTIVE, lw=2, label="Active")
    ax.stairs(ov["passive"], edges, color=PASSIVE, lw=2, label="Passive")
    ax.set_xlabel("Predicted chance a discovery is passive"); ax.set_ylabel("Discoveries")
    ax.set_title("Overlap", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ov_chart = svg(fig)

    e, (lo, hi) = psm["estimate"], psm["ci95"]
    if ci_excludes_zero(psm["ci95"]):
        direction = "less" if e < 0 else "more"
        headline = (f"Artists that simply played next came back {abs(100 * e):.1f} percentage points "
                    f"{direction} often than artists I clicked on in comparable situations "
                    f"(95% CI {100 * lo:+.1f} to {100 * hi:+.1f}).")
    else:
        headline = (f"After comparing like situations, there is no clear difference between passive "
                    f"and active discoveries ({pts(e)}, 95% CI {100 * lo:+.1f} to {100 * hi:+.1f}).")
    share_note = ""
    if psm["share_matched"] < 0.9:
        share_note = (f" This compares the {pct(psm['share_matched'])} of passive discoveries that had a "
                      f"close active counterpart; the rest happened in situations where I almost never "
                      f"clicked a song myself, so there is nothing fair to compare them with.")
    gap_note = ""
    if res["naive"]["estimate"] * e < 0:
        # The sign flipped after adjustment: say so plainly.
        gap_note = (f" The raw gap was {pts(res['naive']['estimate'])}; comparing like situations "
                    f"flipped its sign, so the raw number mostly reflected <em>when and where</em> "
                    f"each kind of discovery happens.")
    elif abs(res["naive"]["estimate"] - e) >= 0.01:
        bigger = abs(res["naive"]["estimate"]) > abs(e)
        gap_note = (f" The raw gap was {pts(res['naive']['estimate'])}; part of it came from "
                    f"<em>when and where</em> passive listening happens, not from the path itself."
                    if bigger else
                    f" The raw gap was {pts(res['naive']['estimate'])}; adjusting for context made "
                    f"the difference larger, not smaller.")
    balance_ok = res["max_abs_smd_after"] < 0.1
    body = f"""
      <p class="lede">{headline}</p>
      <p>Passive means the song started because the previous one ended (autoplay, radio,
      queue); active means I clicked that exact song. Discovery Mode lives in the passive
      lane, which is why this split matters.{gap_note}{share_note}</p>
      <div class="charts"><figure>{est_chart}</figure><figure>{bal_chart}</figure></div>
      <details><summary>Method and diagnostics</summary>
        <p>A logistic model estimated each discovery's chance of being passive from context
        known before the song finished: time of day, weekend, device, shuffle, offline, year, and a
        time trend. Each passive discovery was compared only with active discoveries from the
        same year, on the same kind of device, in the same part of the day, on the same kind of
        day (weekday or weekend), using up to
        {config.PSM_NEIGHBORS} with the closest score within a caliper of {config.PSM_CALIPER_SD} SD
        ({psm['n_matched']:,} of {psm['n_treated']:,} passive discoveries matched).
        Listen length and same-day replays were deliberately left out: they happen
        <em>after</em> the path is set, so adjusting for them would hide part of the effect.</p>
        <p>Balance: {"every covariate ended inside the ±0.1 band" if balance_ok else
        f"the largest remaining standardized difference is {res['max_abs_smd_after']:.2f}, above the 0.1 guideline, so treat the estimate with extra caution"}.
        Confidence intervals come from {config.N_BOOTSTRAP} bootstrap resamples of the whole procedure.</p>
        <figure class="narrow">{ov_chart}</figure>
        <p class="caveat">Matching only balances what was measured. Mood, activity, or a friend's
        recommendation are not in the data, so this is evidence, not proof. The coin-flip experiment
        below is the design that removes that gap.</p>
      </details>"""
    return body, {"psm": e, "psm_ci": psm["ci95"], "naive": res["naive"]["estimate"],
                  "passive_rate": psm["matched_passive_rate"], "active_rate": psm["matched_active_rate"]}


# ---------------------------------------------------------------------------
# Section 4: Experiment
# ---------------------------------------------------------------------------
def section_experiment(res: dict) -> tuple:
    status = res.get("status", "missing")
    treat, ctrl = esc(res.get("treatment_label", "")), esc(res.get("control_label", ""))
    power = res.get("power", {})

    power_html = ""
    if power.get("status") == "ok":
        rows = "".join(
            f"<tr><td>{r['days']}</td><td>{r['mde_artists_per_day']:.2f}</td>"
            f"<td>{'' if r['mde_pct_of_mean'] is None else str(r['mde_pct_of_mean']) + '%'}</td></tr>"
            for r in power["table"])
        fpr = power.get("aa_false_positive_rate")
        power_html = f"""
          <p>Before starting: my history averages {power['mean_new_artists_per_day']:.1f} new artists a
          day. The table shows the smallest true change each experiment length can reliably detect
          (5% false-positive rate, 80% power).</p>
          <div class="table-wrap"><table class="compact">
            <thead><tr><th>Days</th><th>Detectable change (artists/day)</th><th>As % of average</th></tr></thead>
            <tbody>{rows}</tbody></table></div>
          <p>A/A check: across 1,000 fake experiments on past days where nothing changed,
          {pct(fpr, 1)} came out "significant", in line with the 5% the test is designed to allow.</p>"""

    if status == "complete":
        a = res["analysis"]
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        pairs = pd.DataFrame(a["pairs"])
        for r in pairs.itertuples():
            ax.plot([0, 1], [r.control, r.treatment], color=GRID, lw=1)
        ax.plot([0, 1], [a["control_mean"], a["treatment_mean"]], color=ACTIVE, lw=3, marker="o", ms=8)
        ax.set_xticks([0, 1], [res["control_label"], res["treatment_label"]])
        ax.set_xlim(-0.3, 1.3)
        ax.set_ylabel("New artists that day")
        ax.set_title("Each pair of days, and the average", loc="left", fontsize=11)
        ax.grid(axis="x", visible=False)
        chart = svg(fig)
        sig = a["p_value_randomization"] < config.EXPERIMENT_ALPHA
        headline = (f"{treat} changed daily discoveries by {a['mean_diff']:+.2f} artists "
                    f"({a['lift_pct']:+.0f}%; 95% CI {a['ci95'][0]:+.2f} to {a['ci95'][1]:+.2f}, "
                    f"randomization p = {a['p_value_randomization']:.3f})."
                    if sig else
                    f"No clear effect: {treat} days averaged {a['treatment_mean']:.2f} new artists vs "
                    f"{a['control_mean']:.2f} ({a['mean_diff']:+.2f}; 95% CI {a['ci95'][0]:+.2f} to "
                    f"{a['ci95'][1]:+.2f}, randomization p = {a['p_value_randomization']:.3f}).")
        body = f"""<p class="lede">{headline}</p>
          <div class="charts"><figure>{chart}</figure></div>
          <details><summary>Design and power</summary>{power_html}</details>"""
        return body, {"experiment": a}

    state = {
        "not_planned": ("The experiment is designed and ready to start. Create the randomized "
                        "schedule with: python -m src.experiment plan"),
        "not_started": f"Scheduled to run {esc(res.get('schedule_start'))} to {esc(res.get('schedule_end'))}.",
        "in_progress": f"Running: {res.get('days_covered')} of {res.get('days_total')} days are in the data so far.",
    }.get(status, "The experiment has not been run.")
    body = f"""
      <p class="lede">{state}</p>
      <p>The matched comparison above can only adjust for what was recorded. To remove that
      limitation, days are randomized in pairs: a coin flip decides which day of each pair is
      <strong>{treat}</strong> and which is <strong>{ctrl}</strong>. The outcome, new artists that day, comes
      straight from the streaming export, and the analysis follows the assignment even on days
      I forgot (intention to treat).</p>
      {power_html}"""
    return body, {}


# ---------------------------------------------------------------------------
# Section 5: Methods
# ---------------------------------------------------------------------------
def section_methods(meta: dict) -> str:
    return f"""
      <dl class="defs">
        <dt>Set aside</dt><dd>Sleep and background listening: artists whose titles are mostly
          sleep or noise phrases, and every play in a session that ran {config.BACKGROUND_LONG_SESSION_MIN // 60}+
          hours (or {config.BACKGROUND_NIGHT_SESSION_MIN}+ minutes starting at night) with skips or clicks on
          under {config.BACKGROUND_MAX_ACTION_RATE:.0%} of tracks. {meta['n_foreground']:,} of
          {meta['n_plays']:,} plays remain for analysis.</dd>
        <dt>Stream</dt><dd>A play of at least {config.MIN_PLAY_MS // 1000} seconds. Shorter plays count as skips.</dd>
        <dt>Discovery</dt><dd>The first stream of an artist, excluding the first {config.BURN_IN_DAYS} days of
          history, where most "new" artists were already familiar.</dd>
        <dt>Return (conversion)</dt><dd>Any stream of that artist on a later calendar day within
          {config.RETENTION_WINDOW_DAYS} days. Discoveries in the final {config.RETENTION_WINDOW_DAYS}
          days are counted but not labeled, because their window is not over.</dd>
        <dt>Data</dt><dd>{meta['n_plays']:,} music plays from {esc(meta['start'])} to {esc(meta['end'])},
          loaded from Spotify's {esc(', '.join(meta['formats']))} export. Personal identifiers
          (IP address, user agent) were dropped on load. Times are in {esc(config.LOCAL_TIMEZONE)}.</dd>
        <dt>Pipeline</dt><dd>Python loads the JSON; SQL models (staging, intermediate, marts) build the
          tables; {meta['n_checks']} automated data-quality checks recompute key numbers a second way
          before anything is published.</dd>
      </dl>
      <p class="caveat">Limitations: one listener, so none of this generalizes to Spotify's users. Artists
      are matched by name because the export has no artist ID. The export says how a song started, not
      which playlist or algorithm served it, so this measures discovery behavior, not Spotify's
      recommendation systems.</p>"""


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#FFFFFF;--muted:#B3B3B3;--line:#2A2A2A;--bg:#121212;--panel:#181818;
  --green:#1DB954;--passive:#F59B23;color-scheme:dark;
  --sans:Inter,"Segoe UI","Helvetica Neue",Arial,system-ui,sans-serif}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%;background:var(--bg)}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 var(--sans);
  padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}
main{max-width:1080px;margin:0 auto;padding:48px 24px 80px}
header.hero{padding:8px 0 40px;margin-bottom:8px;border-bottom:1px solid var(--line)}
.byline{color:var(--muted);font-size:14px;margin:0 0 18px}
h1{font:800 clamp(34px,6vw,60px)/1.05 var(--sans);letter-spacing:-.03em;margin:0 0 18px;max-width:18ch}
h1 .pct{color:var(--green)}
.hero .sub{font-size:18px;color:var(--muted);max-width:62ch;margin:0}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:32px 0 0}
.facts div{background:var(--panel);border-radius:8px;padding:14px 16px}
.facts dt{font-size:13px;color:var(--muted)}
.facts dd{margin:4px 0 0;font:700 28px/1.2 var(--sans);letter-spacing:-.02em;font-variant-numeric:tabular-nums}
section{padding:44px 0;border-bottom:1px solid var(--line)}
h2{font:700 28px/1.2 var(--sans);letter-spacing:-.02em;margin:0 0 14px;max-width:32ch}
h3{font:700 16px/1.3 var(--sans);margin:28px 0 8px}
p{max-width:70ch;margin:0 0 14px;color:#E5E5E5}
.lede{font-size:19px;line-height:1.5;color:var(--ink)}
.lede strong{color:var(--green);font-variant-numeric:tabular-nums}
em{color:var(--ink)}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:20px;margin:24px 0}
.charts.single{grid-template-columns:minmax(0,760px)}
figure{margin:0;background:var(--panel);border-radius:8px;padding:16px}
figure.narrow{max-width:460px;margin:16px 0}
figcaption{font-size:13px;color:var(--muted);margin-top:8px}
svg.chart{width:100%;height:auto;display:block}
details{margin-top:8px;border-left:3px solid var(--green);padding-left:16px}
summary{cursor:pointer;font-weight:700;padding:6px 0}
summary:focus-visible,a:focus-visible{outline:2px solid var(--green);outline-offset:3px}
.table-wrap{overflow-x:auto;margin:12px 0}
table{border-collapse:collapse;font-size:14px;min-width:520px;font-variant-numeric:tabular-nums}
table.compact{min-width:360px}
th,td{text-align:left;padding:8px 16px 8px 0;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted)}
ol.stuck{list-style:none;padding:0;margin:0;max-width:560px;counter-reset:s}
ol.stuck li{display:flex;justify-content:space-between;gap:16px;padding:9px 0;border-bottom:1px solid var(--line);counter-increment:s}
ol.stuck li span:first-child::before{content:counter(s);display:inline-block;width:1.8em;color:var(--green);font-weight:700}
.num{color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.notice{background:#2A2112;border-left:3px solid var(--passive);padding:12px 16px;border-radius:4px}
.caveat{color:var(--muted);font-size:14px}
.banner{background:var(--green);color:#000;font-weight:700;padding:10px 24px;font-size:14px;text-align:center}
dl.defs{display:grid;grid-template-columns:minmax(120px,180px) 1fr;gap:10px 24px;max-width:860px}
dl.defs dt{font-weight:700}
dl.defs dd{margin:0;color:#E5E5E5}
footer{color:var(--muted);font-size:13px;padding-top:24px}
@media (max-width:640px){main{padding:32px 18px 64px}dl.defs{grid-template-columns:1fr}
  dl.defs dd{margin-bottom:8px}.lede{font-size:17px}}
"""


def build(db_path: Path = config.DB_PATH, output_dir: Path = config.OUTPUT_DIR,
          is_sample: bool = False, n_checks: int = 0) -> Path:
    global _chart_counter
    _chart_counter = 0
    with sqlite3.connect(db_path) as con:
        monthly = pd.read_sql("SELECT * FROM mart_monthly_kpis", con)
        events = pd.read_sql("SELECT * FROM mart_discovery_events", con)
        meta_row = con.execute("SELECT MIN(listen_date), MAX(listen_date), COUNT(*) FROM stg_streams").fetchone()
        n_foreground = con.execute("SELECT COUNT(*) FROM stg_foreground_streams").fetchone()[0]
        formats = [r[0] for r in con.execute("SELECT DISTINCT source_format FROM stg_streams")]
    meta = {"start": meta_row[0], "end": meta_row[1], "n_plays": meta_row[2], "n_foreground": n_foreground,
            "formats": formats, "n_checks": n_checks}

    model = load_json(output_dir / "model_results.json")
    causal = load_json(output_dir / "causal_results.json")
    experiment = load_json(output_dir / "experiment_results.json")

    with sqlite3.connect(db_path) as con:
        bg_html, bg = section_background(con, monthly)
        # Review file: every flagged artist and why, for checking the rules.
        # (outputs/ is git-ignored, so this personal list stays local.)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.read_sql("SELECT * FROM int_background_artists ORDER BY hours DESC", con).to_csv(
            output_dir / "background_artists.csv", index=False)
    kpi_html, kpi = section_kpis(monthly, events)
    model_html, mod = section_model(model)
    causal_html, cau = section_causal(causal)
    exp_html, _ = section_experiment(experiment)

    # The hero states the main finding as a sentence; the facts row backs it up.
    facts = [("Set aside as sleep / background", pct(bg.get("background_share"))),
             ("New artists met", f"{kpi['n_discoveries']:,}"),
             (f"Came back in {config.RETENTION_WINDOW_DAYS} days", pct(kpi["conversion_rate"]))]
    if mod:
        facts.append(("Model AUC on future data", f"{mod['auc']:.2f}"))
    if cau:
        facts.append(("Passive vs active, matched", pts(cau["psm"])))
    facts_html = "".join(f"<div><dt>{esc(k)}</dt><dd>{esc(v)}</dd></div>" for k, v in facts)

    # The headline percentage is set in green; everything else is escaped text.
    title_html = (f'<span class="pct">{esc(pct(kpi["conversion_rate"]))}</span> of the artists I '
                  f'discover earn a return visit' if kpi["conversion_rate"] is not None
                  else "How my music discovery converts")
    title = re.sub("<[^>]+>", "", title_html)
    banner = ('<div class="banner">Sample data: these numbers come from a synthetic listening '
              'history, not a real person.</div>') if is_sample else ""

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Did discovery stick? | {esc(config.AUTHOR_NAME)}</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🎧</text></svg>">
<style>{CSS}</style>
</head>
<body>
{banner}
<main>
  <header class="hero">
    <p class="byline">{esc(config.AUTHOR_NAME)}, measuring music discovery in my own Spotify
      history from {esc(meta['start'])} to {esc(meta['end'])}</p>
    <h1>{title_html}</h1>
    <p class="sub">A measurement study in the spirit of Discovery Mode: define what a successful
      discovery is, predict it from the first listen, test whether the way a song reaches me
      changes it, and set up a randomized experiment to check.</p>
    <dl class="facts">{facts_html}</dl>
  </header>

  <section aria-labelledby="s0"><h2 id="s0">First, setting aside what I slept through</h2>{bg_html}</section>
  <section aria-labelledby="s1"><h2 id="s1">How much am I discovering, and how much sticks?</h2>{kpi_html}</section>
  <section aria-labelledby="s2"><h2 id="s2">Can the first listen predict which artists stick?</h2>{model_html}</section>
  <section aria-labelledby="s3"><h2 id="s3">Does it matter how I find an artist?</h2>{causal_html}</section>
  <section aria-labelledby="s4"><h2 id="s4">Testing it with a coin flip</h2>{exp_html}</section>
  <section aria-labelledby="s5"><h2 id="s5">How this was measured</h2>{section_methods(meta)}</section>

  <footer>Built with Python, SQL (SQLite), scikit-learn, and matplotlib. Generated
    {datetime.now().strftime('%B %d, %Y')}.</footer>
</main>
</body>
</html>"""

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")

    # Headline numbers in one file, handy for the README and resume.
    summary = {"background": bg, "kpis": kpi, "model": mod, "causal": cau, "experiment_status": experiment.get("status")}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"  wrote {out}")
    return out


if __name__ == "__main__":
    build()
