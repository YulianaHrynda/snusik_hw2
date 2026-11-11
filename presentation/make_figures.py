#!/usr/bin/env python
"""Render the presentation charts from the gold tables (local run: catalog spark_catalog).

    python presentation/make_figures.py
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT = os.path.join(REPO, "presentation", "figures")
G = "spark_catalog.airbnb_gold"

# reference palette (light): text tokens, neutral for the baseline, ordinal blue for model versions,
# categorical slots for the three cities
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BASELINE = "#a8a7a2"
V1, V2 = "#86b6ef", "#1c5cab"
SHORT, MONTHLY = "#eb6834", "#1baf7a"   # stay type: categorical slots 2 and 3 (never the version blues)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 10, "font.family": "DejaVu Sans",
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", name)


def bar_labels(ax, bars, fmt="${:,.0f}", inside=False):
    for b in bars:
        h = b.get_height()
        if np.isnan(h):
            continue
        ax.annotate(fmt.format(h), (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK)


def rounded_bar(ax, x, h, w, color, label=None):
    return ax.bar(x, h, w, color=color, label=label, edgecolor=SURFACE, linewidth=2, zorder=3)


def main():
    from run_local import spark_session
    spark = spark_session()
    q = lambda sql: spark.sql(sql).toPandas()

    # 1. price by stay type and city (EDA headline) ------------------------------------------------
    seg = q(f"""SELECT city_name, monthly_stay, percentile(price, 0.5) AS med, count(*) AS n
                FROM {G}.listings WHERE has_valid_price GROUP BY 1, 2""")
    share = q(f"""SELECT city_name, avg(CAST(monthly_stay AS DOUBLE)) AS monthly FROM {G}.listings GROUP BY 1""")
    cities = ["Boston", "Jersey City", "New York City"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    x = np.arange(len(cities))
    for i, (flag, label, color) in enumerate([(False, "short stay (< 28 nights)", SHORT), (True, "monthly (28+ nights)", MONTHLY)]):
        vals = [seg[(seg.city_name == c) & (seg.monthly_stay == flag)].med.iloc[0] for c in cities]
        bar_labels(ax, rounded_bar(ax, x + (i - 0.5) * 0.36, vals, 0.34, color, label))
    ax.set_xticks(x, cities)
    ax.set_title("Median price per night")
    ax.set_ylim(0, 420)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8.5, ncol=2)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax = axes[1]
    vals = [100 * share[share.city_name == c].monthly.iloc[0] for c in cities]
    bar_labels(ax, rounded_bar(ax, x, vals, 0.5, MONTHLY), fmt="{:.0f}%")
    ax.set_xticks(x, ["Boston", "Jersey\nCity", "New York\nCity"])
    ax.set_ylim(0, 100)
    ax.set_title("Monthly-stay listings, % of all")
    fig.tight_layout()
    save(fig, "stay_type.png")

    # 2. A/B test MAE: baseline vs v1 -------------------------------------------------------------
    m = q(f"""SELECT model_version, city, split, n, model_mae, baseline_mae, baseline_local_mae
              FROM {G}.model_metrics WHERE scope = 'trainable' AND split IN ('test', 'all')""")
    names = {"boston": "Boston (A)", "jersey-city": "Jersey City (B)", "ALL": "A + B"}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    keys = ["boston", "jersey-city", "ALL"]
    x = np.arange(3)
    b = [m[(m.model_version == "v1") & (m.city == k) & (m.split == "test")].baseline_mae.iloc[0] for k in keys]
    v = [m[(m.model_version == "v1") & (m.city == k) & (m.split == "test")].model_mae.iloc[0] for k in keys]
    bar_labels(ax, rounded_bar(ax, x - 0.19, b, 0.36, BASELINE, "baseline (median by room type × guests)"))
    bar_labels(ax, rounded_bar(ax, x + 0.19, v, 0.36, V1, "model v1"))
    ax.set_xticks(x, [names[k] for k in keys])
    ax.set_title("Test-set MAE, USD per night (lower is better)")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_ylim(0, max(b) * 1.25)
    fig.tight_layout()
    save(fig, "ab_mae.png")

    # 3. permutation importance (v1) ---------------------------------------------------------------
    card = json.load(open(os.path.join(REPO, "models", "v1", "card.json")))
    imp = pd.DataFrame(card["feature_importance"]).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.barh(imp.feature, imp.mae_increase, color=V1, height=0.6, zorder=3)
    for yy, val in enumerate(imp.mae_increase):
        ax.annotate(f"+${val:.0f}", (val, yy), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.grid(axis="x"); ax.grid(axis="y", visible=False)
    ax.set_title("What v1 relies on: test MAE increase when shuffled")
    ax.set_xlim(0, imp.mae_increase.max() * 1.18)
    fig.tight_layout()
    save(fig, "importance.png")

    # 4. NYC first run by borough ----------------------------------------------------------------
    bo = q(f"""SELECT l.neighbourhood_group AS borough, count(*) AS n,
                      percentile(p.actual_price, 0.5) AS actual, percentile(p.predicted_price, 0.5) AS predicted,
                      avg(p.abs_error) AS mae
               FROM {G}.price_predictions p JOIN {G}.listings l USING (city, listing_id)
               WHERE p.city = 'new-york-city' AND p.model_version = 'v1' AND p.is_trainable
               GROUP BY 1 ORDER BY actual""")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(bo))
    bar_labels(ax, rounded_bar(ax, x - 0.19, bo.actual, 0.36, BASELINE, "actual median price"))
    bar_labels(ax, rounded_bar(ax, x + 0.19, bo.predicted, 0.36, V1, "v1 predicted median"))
    ax.set_xticks(x, [f"{b}\n(MAE ${mae:.0f})" for b, mae in zip(bo.borough, bo.mae)])
    ax.set_title("NYC first run: fine in the outer boroughs, Manhattan priced like Jersey City")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, bo.actual.max() * 1.25)
    fig.tight_layout()
    save(fig, "nyc_borough_v1.png")

    # 5. side-by-side results ---------------------------------------------------------------------
    def get(ver, city, split, col="model_mae"):
        r = m[(m.model_version == ver) & (m.city == city) & (m.split == split)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    groups = ["A + B (test)", "New York City (test)"]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(2)
    series = [
        ("baseline (A+B medians)", BASELINE, [get("v1", "ALL", "test", "baseline_mae"), get("v1", "new-york-city", "test", "baseline_mae")]),
        ("model v1 (first run)", V1, [get("v1", "ALL", "test"), get("v1", "new-york-city", "test")]),
        ("model v2 (after changes)", V2, [get("v2", "ALL", "test"), get("v2", "new-york-city", "test")]),
    ]
    for i, (label, color, vals) in enumerate(series):
        bar_labels(ax, rounded_bar(ax, x + (i - 1) * 0.26, vals, 0.24, color, label))
    ax.set_xticks(x, groups)
    ax.set_title("MAE, USD per night: at home vs the new client")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_ylim(0, np.nanmax([v for _, _, vals in series for v in vals]) * 1.25)
    fig.tight_layout()
    save(fig, "results.png")

    # 6. predicted vs actual on NYC test, v1 vs v2 -----------------------------------------------
    pv = q(f"""SELECT model_version, actual_price, predicted_price FROM {G}.price_predictions
               WHERE city = 'new-york-city' AND is_trainable AND split = 'test' AND model_version IN ('v1', 'v2')""")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharex=True, sharey=True)
    lim = (25, 3000)
    for ax, ver, color in zip(axes, ["v1", "v2"], [V1, V2]):
        d = pv[pv.model_version == ver]
        ax.scatter(d.actual_price, d.predicted_price, s=5, alpha=0.25, color=color, linewidths=0, zorder=3)
        ax.plot(lim, lim, color=INK2, linewidth=1, linestyle="--", zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
        mae = (d.predicted_price - d.actual_price).abs().mean()
        ax.set_title(f"{ver}: MAE ${mae:.0f}")
        ax.set_xlabel("actual price, USD (log)")
        ax.grid(axis="both")
        for a in (ax.xaxis, ax.yaxis):
            a.set_major_formatter(matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    axes[0].set_ylabel("predicted price, USD (log)")
    fig.suptitle("New York City test listings: dashed line = perfect prediction", x=0.01, ha="left",
                 fontsize=10, color=INK2)
    fig.tight_layout()
    save(fig, "nyc_scatter.png")


if __name__ == "__main__":
    main()
