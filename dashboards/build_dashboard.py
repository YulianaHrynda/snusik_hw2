#!/usr/bin/env python
from __future__ import annotations

import json
import os

CATALOG = "workspace"
GOLD = f"{CATALOG}.airbnb_gold"
TRAIN_CITIES = "('boston', 'jersey-city')"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "airbnb_us_northeast.lvdash.json")

TITLES = {
    "overview": "Boston has the priciest typical night ($253 median); New York is a monthly-rental market (81% of listings need 30+ nights)",
    "compare": "An entire home costs the same in both cities (~$300); Jersey City is cheaper only for private rooms (−32%) and small units",
    "model": "41% better than the baseline at home, but the first New York run under-priced by $129 a night; v2 cuts NYC error from $144 to $115",
}
SUBTITLES = {
    "overview": "Pick a city. Location drives price inside a city (Boston: Longwood, North End and Downtown ~$360+, East Boston ~$220). "
                "Multi-listing hosts run 73–80% of listings in Boston and Jersey City, 57% in New York. Prices are USD per night; "
                "medians use listings with a valid price. New York (city C) was added after the model was built.",
    "compare": "Monthly rentals (28+ night minimum) are ~42% of listings in both cities and cost about half as much per night. "
               "Boston's premium sits in 1–3 guest units ($211 vs $108 for 2 guests) and disappears for 6+ guests.",
    "model": "Test listings only (New York: the same 20% hash split, never trained on). Baseline = median A+B training price for the same "
             "room type and guests. v1 = week-1 model; v2 = after the city C changes (market comps, no city-specific features), still "
             "trained on Boston + Jersey City; v2_abc = reference retrained with NYC data. Negative error = under-priced: the biggest "
             "misses are Manhattan short stays and hotels, and a few luxury homes in Back Bay and Ward D.",
}

DATASETS = {
    "city_metrics": ("City metrics", f"""
SELECT city, city_name, snapshot_date, listings, priced_listings, hosts, neighbourhoods,
       round(median_price, 0) AS median_price, round(median_price_entire_home, 0) AS median_price_entire_home,
       round(median_price_private_room, 0) AS median_price_private_room,
       median_availability_365, round(100 * blocked_calendar_share, 1) AS blocked_calendar_pct,
       round(avg_review_score, 2) AS avg_review_score,
       round(100 * multi_listing_host_share, 1) AS multi_listing_host_pct,
       round(100 * superhost_share, 1) AS superhost_pct,
       round(100 * monthly_stay_share, 1) AS monthly_stay_pct,
       round(listings / hosts, 2) AS listings_per_host
FROM {GOLD}.city_metrics"""),
    "room_mix": ("Room-type mix", f"""
SELECT city_name, room_type, count(*) AS listings,
       round(100 * count(*) / sum(count(*)) OVER (PARTITION BY city_name), 1) AS share_pct,
       round(percentile(CASE WHEN has_valid_price THEN price END, 0.5), 0) AS median_price
FROM {GOLD}.listings
GROUP BY city_name, room_type"""),
    "neighbourhoods": ("Neighbourhoods", f"""
SELECT city_name, coalesce(neighbourhood_group, '—') AS neighbourhood_group, neighbourhood, listings, priced_listings,
       round(median_price, 0) AS median_price,
       round(median_price_entire_home, 0) AS median_price_entire_home,
       round(100 * entire_home_share, 1) AS entire_home_pct, round(avg_review_score, 2) AS avg_review_score,
       median_availability_365,
       row_number() OVER (PARTITION BY city ORDER BY median_price DESC) AS price_rank
FROM {GOLD}.neighbourhood_metrics
WHERE priced_listings >= 10"""),
    "top_neighbourhoods": ("Top 20 neighbourhoods by price", f"""
SELECT city_name, neighbourhood, round(median_price, 0) AS median_price, priced_listings
FROM (SELECT *, row_number() OVER (PARTITION BY city ORDER BY median_price DESC) AS rk
      FROM {GOLD}.neighbourhood_metrics WHERE priced_listings >= 10)
WHERE rk <= 20"""),
    "compare_table": ("A vs B side by side", f"""
WITH m AS (SELECT * FROM {GOLD}.city_metrics WHERE city IN {TRAIN_CITIES}),
b AS (SELECT * FROM m WHERE city = 'boston'), j AS (SELECT * FROM m WHERE city = 'jersey-city'),
rows AS (
  SELECT 1 AS ord, 'Listings' AS metric, b.listings AS boston, j.listings AS jersey_city FROM b, j UNION ALL
  SELECT 2, 'Median price, $', b.median_price, j.median_price FROM b, j UNION ALL
  SELECT 3, 'Median price, entire home, $', b.median_price_entire_home, j.median_price_entire_home FROM b, j UNION ALL
  SELECT 4, 'Median price, private room, $', b.median_price_private_room, j.median_price_private_room FROM b, j UNION ALL
  SELECT 5, 'Median price, short stay (<28 nights), $', b.median_price_short_stay, j.median_price_short_stay FROM b, j UNION ALL
  SELECT 6, 'Median price, monthly stay (28+ nights), $', b.median_price_monthly_stay, j.median_price_monthly_stay FROM b, j UNION ALL
  SELECT 7, 'Monthly-stay listings, %', round(100 * b.monthly_stay_share, 1), round(100 * j.monthly_stay_share, 1) FROM b, j UNION ALL
  SELECT 8, 'Entire homes, %', round(100 * b.entire_home_share, 1), round(100 * j.entire_home_share, 1) FROM b, j UNION ALL
  SELECT 9, 'Median days available (next 365)', b.median_availability_365, j.median_availability_365 FROM b, j UNION ALL
  SELECT 10, 'Average review score', round(b.avg_review_score, 2), round(j.avg_review_score, 2) FROM b, j UNION ALL
  SELECT 11, 'Hosts', b.hosts, j.hosts FROM b, j UNION ALL
  SELECT 12, 'Listings run by multi-listing hosts, %', round(100 * b.multi_listing_host_share, 1), round(100 * j.multi_listing_host_share, 1) FROM b, j UNION ALL
  SELECT 13, 'Superhost listings, %', round(100 * b.superhost_share, 1), round(100 * j.superhost_share, 1) FROM b, j
)
SELECT ord, metric, round(boston, 1) AS boston, round(jersey_city, 1) AS jersey_city,
       round(100 * (jersey_city - boston) / boston, 1) AS jc_vs_boston_pct
FROM rows ORDER BY ord"""),
    "price_by_capacity": ("Median price by capacity", f"""
SELECT city_name, least(accommodates, 10) AS guests, round(percentile(price, 0.5), 0) AS median_price, count(*) AS listings
FROM {GOLD}.listings
WHERE has_valid_price AND city IN {TRAIN_CITIES}
GROUP BY city_name, least(accommodates, 10)"""),
    "price_by_segment": ("Median price by room type and stay", f"""
SELECT city_name, concat(room_type, CASE WHEN monthly_stay THEN ' · monthly' ELSE ' · short stay' END) AS segment,
       round(percentile(price, 0.5), 0) AS median_price, count(*) AS listings
FROM {GOLD}.listings
WHERE has_valid_price AND city IN {TRAIN_CITIES} AND room_type IN ('Entire home/apt', 'Private room')
GROUP BY ALL"""),
    "model_metrics": ("Model vs baseline", f"""
SELECT model_version,
       CASE model_version WHEN 'v1' THEN 'v1 · week 1' WHEN 'v2' THEN 'v2 · after city C'
            WHEN 'v2_abc' THEN 'v2_abc · reference, trained with NYC' ELSE model_version END AS model_label,
       CASE city WHEN 'ALL' THEN 'A + B together' WHEN 'boston' THEN 'Boston'
         WHEN 'jersey-city' THEN 'Jersey City' WHEN 'new-york-city' THEN 'New York City' ELSE city END AS city_name,
       split, n, round(model_mae, 1) AS model_mae, round(baseline_mae, 1) AS baseline_mae,
       round(100 * mae_improvement_pct, 1) AS improvement_pct, round(model_median_ae, 1) AS model_median_ae,
       round(100 * model_mape, 1) AS model_mape_pct, round(model_bias, 1) AS model_bias
FROM {GOLD}.model_metrics
WHERE scope = 'trainable' AND split = 'test' AND model_name <> 'baseline'
ORDER BY model_version, CASE city WHEN 'boston' THEN 1 WHEN 'jersey-city' THEN 2 WHEN 'ALL' THEN 3 ELSE 4 END"""),
    "mae_by_version": ("MAE by version", f"""
SELECT CASE city WHEN 'ALL' THEN 'Boston + Jersey City' ELSE 'New York City' END AS evaluated_on,
       'baseline' AS series, round(baseline_mae, 1) AS mae
FROM {GOLD}.model_metrics
WHERE model_version = 'v1' AND split = 'test' AND scope = 'trainable' AND city IN ('ALL', 'new-york-city')
UNION ALL
SELECT CASE city WHEN 'ALL' THEN 'Boston + Jersey City' ELSE 'New York City' END, model_version, round(model_mae, 1)
FROM {GOLD}.model_metrics
WHERE model_version IN ('v1', 'v2') AND split = 'test' AND scope = 'trainable' AND city IN ('ALL', 'new-york-city')"""),
    "predictions": ("Predictions (evaluation listings)", f"""
SELECT p.model_version, l.city_name, p.listing_id, p.room_type, p.neighbourhood,
       coalesce(l.neighbourhood_group, l.neighbourhood) AS area, p.accommodates,
       p.actual_price, p.predicted_price, p.baseline_price, p.error, p.abs_error,
       round(100 * p.pct_error, 1) AS pct_error, p.city_role, l.listing_url
FROM {GOLD}.price_predictions p JOIN {GOLD}.listings l USING (city, listing_id)
WHERE p.is_trainable AND p.split = 'test'"""),
    "error_by_room": ("Error by room type", """
SELECT model_version, city_name, room_type, count(*) AS listings,
       round(avg(abs_error), 1) AS model_mae, round(avg(abs(baseline_price - actual_price)), 1) AS baseline_mae,
       round(avg(error), 1) AS mean_error
FROM ({predictions}) GROUP BY ALL"""),
    "error_by_neighbourhood": ("Error by neighbourhood", """
SELECT model_version, city_name, area, concat(city_name, ' · ', area) AS city_area, count(*) AS listings,
       round(avg(error), 1) AS mean_error, round(percentile(error, 0.5), 1) AS median_error,
       round(avg(abs_error), 1) AS model_mae
FROM ({predictions}) GROUP BY ALL HAVING count(*) >= 10"""),
}
DATASETS["worst"] = ("50 biggest misses per model version", """
SELECT * FROM ({predictions})
QUALIFY row_number() OVER (PARTITION BY model_version ORDER BY abs_error DESC) <= 50
ORDER BY abs_error DESC""".format(predictions=DATASETS["predictions"][1]))
DATASETS["error_by_room"] = (DATASETS["error_by_room"][0], DATASETS["error_by_room"][1].format(predictions=DATASETS["predictions"][1]))
DATASETS["error_by_neighbourhood"] = (DATASETS["error_by_neighbourhood"][0],
                                      DATASETS["error_by_neighbourhood"][1].format(predictions=DATASETS["predictions"][1]))


def _q(dataset: str, fields: list[tuple[str, str]], disaggregated: bool = True, name: str = "main_query") -> dict:
    return {"name": name, "query": {"datasetName": dataset, "disaggregated": disaggregated,
                                    "fields": [{"name": n, "expression": e} for n, e in fields]}}


def _frame(title: str, description: str = "") -> dict:
    f = {"showTitle": True, "title": title}
    if description:
        f.update(showDescription=True, description=description)
    return f


def _pos(x, y, w, h) -> dict:
    return {"x": x, "y": y, "width": w, "height": h}


def text(name, lines, pos):
    return {"widget": {"name": name, "multilineTextboxSpec": {"lines": lines}}, "position": _pos(*pos)}


def counter(name, dataset, field, title, pos, agg=None):
    expr = f"{agg}(`{field}`)" if agg else f"`{field}`"
    return {"widget": {"name": name, "queries": [_q(dataset, [(field, expr)], disaggregated=agg is None)],
                       "spec": {"version": 2, "widgetType": "counter",
                                "encodings": {"value": {"fieldName": field, "displayName": title}},
                                "frame": _frame(title)}},
            "position": _pos(*pos)}


def chart(name, kind, dataset, x, y, title, pos, color=None, x_type="categorical", y_type="quantitative",
          sort=None, description="", layout=None, labels=None):
    labels = labels or {}
    fields = [(x, f"`{x}`"), (y, f"`{y}`")] + ([(color, f"`{color}`")] if color else [])
    enc = {"x": {"fieldName": x, "scale": {"type": x_type}, "displayName": labels.get(x, x)},
           "y": {"fieldName": y, "scale": {"type": y_type}, "displayName": labels.get(y, y)}}
    if sort:
        axis, by = sort
        enc[axis]["scale"]["sort"] = {"by": by}
    if color:
        enc["color"] = {"fieldName": color, "scale": {"type": "categorical"}, "displayName": labels.get(color, color)}
    spec = {"version": 3, "widgetType": kind, "encodings": enc, "frame": _frame(title, description)}
    if layout:
        spec["mark"] = {"layout": layout}
    return {"widget": {"name": name, "queries": [_q(dataset, fields)], "spec": spec}, "position": _pos(*pos)}


def table(name, dataset, columns, title, pos, description=""):
    cols = [{"fieldName": c, "displayName": label} for c, label in columns]
    return {"widget": {"name": name, "queries": [_q(dataset, [(c, f"`{c}`") for c, _ in columns])],
                       "spec": {"version": 2, "widgetType": "table", "encodings": {"columns": cols},
                                "frame": _frame(title, description)}},
            "position": _pos(*pos)}


def single_filter(name, datasets, field, title, pos, default=None):
    queries = [_q(ds, [(field, f"`{field}`"), (f"{field}_associativity", "COUNT_IF(`associative_filter_predicate_group`)")],
                  disaggregated=False, name=f"f_{name}_{ds}") for ds in datasets]
    spec = {"version": 2, "widgetType": "filter-single-select",
            "encodings": {"fields": [{"fieldName": field, "displayName": title, "queryName": q["name"]} for q in queries]},
            "frame": _frame(title)}
    if default is not None:
        spec["selection"] = {"defaultSelection": {"values": {"dataType": "STRING", "values": [{"value": default}]}}}
    return {"widget": {"name": name, "queries": queries, "spec": spec}, "position": _pos(*pos)}


def page(name, layout):
    for item in layout:
        if not item["widget"]["name"].endswith("_title"):
            item["position"]["y"] += HEADER_HEIGHT - 2
    return {"name": name, "displayName": {"overview": "City overview", "compare": "Cross-city comparison",
                                          "model": "Model"}[name],
            "pageType": "PAGE_TYPE_CANVAS", "layout": layout}


HEADER_HEIGHT = 3


def header(key):
    lines = [f"## {TITLES[key]}\n"]
    if SUBTITLES[key]:
        lines.append(SUBTITLES[key])
    return text(f"{key}_title", lines, (0, 0, 6, HEADER_HEIGHT))


overview = page("overview", [
    header("overview"),
    single_filter("city_filter", ["city_metrics", "room_mix", "neighbourhoods", "top_neighbourhoods"], "city_name", "City",
                  (0, 2, 2, 2), default="Boston"),
    counter("c_listings", "city_metrics", "listings", "Listings", (2, 2, 1, 2)),
    counter("c_median", "city_metrics", "median_price", "Median price per night, $", (3, 2, 1, 2)),
    counter("c_avail", "city_metrics", "median_availability_365", "Median days available (next 365)", (4, 2, 1, 2)),
    counter("c_score", "city_metrics", "avg_review_score", "Average review score (of 5)", (5, 2, 1, 2)),
    counter("c_hosts", "city_metrics", "hosts", "Hosts", (2, 4, 1, 2)),
    counter("c_multi", "city_metrics", "multi_listing_host_pct", "Listings by multi-listing hosts, %", (3, 4, 1, 2)),
    counter("c_monthly", "city_metrics", "monthly_stay_pct", "Monthly-stay listings, %", (4, 4, 1, 2)),
    counter("c_blocked", "city_metrics", "blocked_calendar_pct", "Fully blocked calendars, %", (5, 4, 1, 2)),
    chart("room_mix_bar", "bar", "room_mix", "room_type", "listings", "Room-type mix", (0, 4, 2, 6),
          labels={"room_type": "Room type", "listings": "Listings"}),
    chart("nbhd_price_bar", "bar", "top_neighbourhoods", "median_price", "neighbourhood",
          "Top 20 neighbourhoods by median price (≥10 priced listings)", (2, 6, 4, 8), x_type="quantitative",
          y_type="categorical", sort=("y", "x-reversed"),
          labels={"median_price": "Median price, $", "neighbourhood": "Neighbourhood"}),
    chart("room_price_bar", "bar", "room_mix", "room_type", "median_price", "Median price by room type, $", (0, 10, 2, 4),
          labels={"room_type": "Room type", "median_price": "Median price, $"}),
    table("nbhd_table", "neighbourhoods",
          [("neighbourhood_group", "Borough / area"), ("neighbourhood", "Neighbourhood"), ("listings", "Listings"),
           ("median_price", "Median price, $"),
           ("median_price_entire_home", "Median entire home, $"), ("entire_home_pct", "Entire homes, %"),
           ("avg_review_score", "Avg review score"), ("median_availability_365", "Median days available")],
          "Neighbourhood details", (0, 14, 6, 7)),
])

compare = page("compare", [
    header("compare"),
    table("compare_table", "compare_table",
          [("metric", "Metric"), ("boston", "Boston (A)"), ("jersey_city", "Jersey City (B)"),
           ("jc_vs_boston_pct", "Jersey City vs Boston, %")], "Side by side", (0, 2, 3, 9)),
    chart("capacity_line", "line", "price_by_capacity", "guests", "median_price",
          "Median price by number of guests (10 = 10+)", (3, 2, 3, 5), color="city_name", x_type="quantitative",
          labels={"guests": "Guests", "median_price": "Median price, $", "city_name": "City"}),
    chart("segment_bar", "bar", "price_by_segment", "segment", "median_price",
          "Median price by room type and stay length", (3, 7, 3, 6), color="city_name", layout="group",
          labels={"segment": "Segment", "median_price": "Median price, $", "city_name": "City"}),
    chart("mix_bar", "bar", "room_mix", "city_name", "share_pct", "Room-type mix, % of listings", (0, 11, 3, 5),
          color="room_type", labels={"city_name": "City", "share_pct": "% of listings", "room_type": "Room type"}),
])

model = page("model", [
    header("model"),
    chart("version_bar", "bar", "mae_by_version", "evaluated_on", "mae",
          "Test MAE, $: baseline vs v1 vs v2 — at home and in the new city", (0, 2, 2, 5), color="series", layout="group",
          labels={"evaluated_on": "Evaluated on", "mae": "MAE, $", "series": "Model"}),
    table("metrics_table", "model_metrics",
          [("model_label", "Model"), ("city_name", "City"), ("n", "Test listings"), ("baseline_mae", "Baseline MAE, $"),
           ("model_mae", "Model MAE, $"), ("improvement_pct", "Improvement, %"), ("model_median_ae", "Median abs. error, $"),
           ("model_mape_pct", "MAPE, %"), ("model_bias", "Mean error (bias), $")],
          "Test-set error by model version and city", (2, 2, 4, 5)),
    single_filter("version_filter", ["predictions", "error_by_room", "error_by_neighbourhood", "worst"],
                  "model_version", "Model version (charts below)", (0, 7, 2, 2), default="v2"),
    chart("pred_scatter", "scatter", "predictions", "actual_price", "predicted_price",
          "Predicted vs actual price (test listings)", (0, 9, 3, 7), color="city_name", x_type="quantitative",
          labels={"actual_price": "Actual price, $", "predicted_price": "Predicted price, $", "city_name": "City"}),
    chart("room_err_bar", "bar", "error_by_room", "room_type", "model_mae", "Model MAE by room type, $",
          (3, 9, 3, 7), color="city_name", layout="group",
          labels={"room_type": "Room type", "model_mae": "MAE, $", "city_name": "City"}),
    chart("nbhd_err_bar", "bar", "error_by_neighbourhood", "mean_error", "city_area",
          "Mean error by area — neighbourhood (Boston, Jersey City) or borough (New York); negative = under-priced",
          (0, 16, 6, 8), x_type="quantitative", y_type="categorical", sort=("y", "x"), color="city_name",
          labels={"mean_error": "Mean error, $", "city_area": "Area", "city_name": "City"}),
    table("worst_table", "worst",
          [("city_name", "City"), ("listing_id", "Listing"), ("room_type", "Room type"), ("neighbourhood", "Neighbourhood"),
           ("accommodates", "Guests"), ("actual_price", "Actual, $"), ("predicted_price", "Predicted, $"),
           ("error", "Error, $"), ("pct_error", "Error, %"), ("listing_url", "URL")],
          "The 50 listings the model mis-prices most (negative error = under-priced)", (0, 24, 6, 8)),
])


def build() -> dict:
    return {
        "datasets": [{"name": k, "displayName": label, "queryLines": [l + "\n" for l in sql.strip().splitlines()]}
                     for k, (label, sql) in DATASETS.items()],
        "pages": [overview, compare, model],
        "uiSettings": {"theme": {"widgetHeaderAlignment": "ALIGNMENT_UNSPECIFIED"}},
    }


if __name__ == "__main__":
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=2, ensure_ascii=False)
        f.write("\n")
    print("wrote", OUT)
