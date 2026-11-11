# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — ingest one city
# MAGIC Downloads the city's pinned Inside Airbnb snapshot into the landing volume (if not there yet)
# MAGIC and loads it into `airbnb_bronze.listings_raw` with every column as a string, replacing only
# MAGIC this city's partition.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from airbnb.bronze import ensure_raw_file, read_listings_csv, to_bronze
from airbnb.tables import write_city_partition
from airbnb.config import get_city, lakehouse_from, read_params
from airbnb.runlog import log_step

params = read_params(dbutils, city="boston", snapshot_date="", catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)
city = get_city(params["city"], params["snapshot_date"])
print(city)

# COMMAND ----------

path, downloaded = ensure_raw_file(lh, city)
print(("downloaded " if downloaded else "already landed ") + path)

raw = read_listings_csv(spark, path)
bronze = to_bronze(raw, city, path)
rows = bronze.count()
if rows == 0:
    raise ValueError(f"{path} produced no rows")

target = lh.table("bronze", "listings_raw")
write_city_partition(bronze, target, city.key)
log_step(spark, lh, city.key, city.snapshot_date, "bronze", None, rows,
         {"file": path, "downloaded": downloaded, "columns": len(raw.columns)})
print(f"{target}: {rows} rows, {len(raw.columns)} source columns for {city.key}")

# COMMAND ----------

dbutils.notebook.exit(f'{{"city": "{city.key}", "bronze_rows": {rows}}}')
