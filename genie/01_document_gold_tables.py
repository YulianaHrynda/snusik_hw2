# Databricks notebook source
# MAGIC %md
# MAGIC # Document the gold tables for Genie
# MAGIC Genie reads Unity Catalog table and column comments when it writes SQL. This notebook puts
# MAGIC the business meaning of every gold column (units, sign conventions, which rows to filter)
# MAGIC into the catalog. Run it once after the gold tables exist; comments survive the per-city
# MAGIC overwrites of the pipeline.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "src")))

from airbnb.config import lakehouse_from, read_params

params = read_params(dbutils, catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)

COMMENTS = {
    "listings": ("One row per Airbnb listing per city at the snapshot date (Inside Airbnb). Prices in USD per night.", {
        "city": "City key: 'boston' (city A), 'jersey-city' (city B), 'new-york-city' (city C, new client)",
        "city_name": "City display name",
        "neighbourhood": "Neighbourhood (Jersey City: council ward)",
        "room_type": "'Entire home/apt', 'Private room', 'Hotel room' or 'Shared room'",
        "accommodates": "Maximum number of guests",
        "price": "Advertised price per night in USD. Use only rows with has_valid_price = true for statistics",
        "price_class": "normal | extreme (above the city's 99th percentile, kept) | implausible (>= $750 per guest, excluded) | missing",
        "has_valid_price": "True when price exists and is not implausible: use this filter for every price statistic",
        "price_per_guest": "price / accommodates, USD",
        "monthly_stay": "True when the minimum stay is 28+ nights (monthly rental; much lower nightly price)",
        "availability_365": "Days available to book in the next 365 days at the snapshot",
        "calendar_blocked": "True when availability_365 = 0 (cannot be booked)",
        "review_scores_rating": "Average guest rating, 0-5; NULL when the listing has no reviews",
        "host_listings_count": "Number of listings the host has in this city",
        "is_stale": "Record carried over from the previous scrape (often without a price)",
    }),
    "city_metrics": ("One row per city: KPIs computed over gold.listings. Medians use valid prices only. Shares are 0-1.", {
        "median_price": "Median price per night, USD (valid prices only)",
        "entire_home_share": "Share of listings that are entire homes, 0-1",
        "multi_listing_host_share": "Share of listings whose host has more than one listing, 0-1",
        "avg_review_score": "Average review_scores_rating of reviewed listings, 0-5",
    }),
    "neighbourhood_metrics": ("One row per city and neighbourhood. Medians use valid prices only.", {
        "median_price": "Median price per night, USD",
        "price_rank_in_city": "1 = most expensive neighbourhood of the city by median price",
    }),
    "price_predictions": ("Price model predictions for every listing, one row per model_version and listing. "
                          "Join to listings on (city, listing_id).", {
        "model_version": "'v1' = week-1 model trained on Boston + Jersey City; 'v2' = model after the city C changes",
        "split": "'train' or 'test' (fixed hash split). Evaluate training cities on split = 'test' only",
        "city_role": "'training' if the model was trained on this city, 'new' otherwise (all rows are out-of-sample)",
        "is_trainable": "True when actual_price is valid; only these rows can be evaluated",
        "actual_price": "Advertised price per night, USD",
        "predicted_price": "Model's suggested price per night, USD",
        "baseline_price": "Baseline: median training price for the same room type and accommodates, USD",
        "error": "predicted_price - actual_price. Negative = model UNDER-prices, positive = OVER-prices",
        "abs_error": "Absolute error, USD (MAE = average of this)",
        "pct_error": "error / actual_price",
    }),
    "model_metrics": ("Model and baseline error per model_version, city ('ALL' = A and B together), split and scope.", {
        "split": "'test' = held-out 20% of training cities; 'all' = every listing of a new city; 'train'; 'cv'",
        "scope": "'trainable' = valid prices (main metric); 'all_priced' also includes implausible prices",
        "model_mae": "Mean absolute error of the model, USD",
        "baseline_mae": "Mean absolute error of the baseline, USD",
        "mae_improvement_pct": "1 - model_mae / baseline_mae (0-1, higher is better)",
        "model_bias": "Mean of predicted - actual, USD (negative = under-pricing on average)",
    }),
}


def sql_str(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")


for table, (table_comment, columns) in COMMENTS.items():
    name = lh.table("gold", table)
    spark.sql(f"COMMENT ON TABLE {name} IS '{sql_str(table_comment)}'")
    for col, comment in columns.items():
        spark.sql(f"ALTER TABLE {name} ALTER COLUMN {col} COMMENT '{sql_str(comment)}'")
    print(f"documented {name}: {len(columns)} columns")
