# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — consumption tables for one city
# MAGIC * `airbnb_gold.listings` — curated listing grain for BI and Genie
# MAGIC * `airbnb_gold.city_metrics`, `airbnb_gold.neighbourhood_metrics` — KPIs
# MAGIC * `airbnb_gold.listing_features` — model features, label and the fixed train/test split

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from airbnb.config import get_city, lakehouse_from, read_params
from airbnb.gold import build_city_metrics, build_listing_features, build_listings, build_neighbourhood_metrics
from airbnb.runlog import log_step
from airbnb.tables import write_city_partition

params = read_params(dbutils, city="boston", snapshot_date="", catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)
city = get_city(params["city"], params["snapshot_date"])

silver = spark.table(lh.table("silver", "listings")).where(f"city = '{city.key}'")
n_silver = silver.count()

# COMMAND ----------

outputs = {
    "listings": build_listings(silver, city.name),
    "city_metrics": build_city_metrics(silver, city.name),
    "neighbourhood_metrics": build_neighbourhood_metrics(silver, city.name),
    "listing_features": build_listing_features(silver),
}
counts = {}
for name, df in outputs.items():
    write_city_partition(df, lh.table("gold", name), city.key, partition=name in ("listings", "listing_features"))
    counts[name] = spark.table(lh.table("gold", name)).where(f"city = '{city.key}'").count()

split = {r["split"]: r["count"] for r in spark.table(lh.table("gold", "listing_features"))
         .where(f"city = '{city.key}' AND is_trainable").groupBy("split").count().collect()}
log_step(spark, lh, city.key, city.snapshot_date, "gold", n_silver, counts["listings"], {**counts, "trainable_split": split})
print(counts, split)

# COMMAND ----------

display(spark.table(lh.table("gold", "city_metrics")).where(f"city = '{city.key}'"))

# COMMAND ----------

dbutils.notebook.exit(f'{{"city": "{city.key}", "gold": {counts["listings"]}}}')
