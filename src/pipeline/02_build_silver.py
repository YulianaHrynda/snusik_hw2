# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — typed and cleaned listings for one city
# MAGIC Applies the cleaning rules of the EDA (section 6): parse money/booleans/dates with
# MAGIC `try_cast`, bathrooms from text, amenities to an array, de-duplicate on `id`, classify price
# MAGIC outliers, drop the leakage columns. Fails loudly if values silently stop parsing.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from airbnb.config import get_city, lakehouse_from, read_params
from airbnb.runlog import log_step
from airbnb.tables import write_city_partition
from airbnb.silver import build_silver

params = read_params(dbutils, city="boston", snapshot_date="", catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)
city = get_city(params["city"], params["snapshot_date"])

# COMMAND ----------

bronze = spark.table(lh.table("bronze", "listings_raw")).where(f"city = '{city.key}'")
silver, report = build_silver(bronze)
print(report)
for w in report["warnings"]:
    print(f"WARNING [{city.key}]: {w}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data-quality gates
# MAGIC * the city must have rows;
# MAGIC * identifiers and prices must parse — more than 1% of present values turning into NULL means
# MAGIC   the source format changed and the run must stop rather than publish wrong numbers.

# COMMAND ----------

if report["rows_out"] == 0:
    raise ValueError(f"No silver rows for {city.key}")
for col in ("id", "price"):
    present = bronze.where(f"`{col}` IS NOT NULL").count()
    failed = report["parse_failures"][col]
    if present and failed / present > 0.01:
        raise ValueError(f"{failed} of {present} non-null '{col}' values failed to parse for {city.key}")

target = lh.table("silver", "listings")
write_city_partition(silver, target, city.key)
log_step(spark, lh, city.key, city.snapshot_date, "silver", report["rows_in"], report["rows_out"], report)

display(spark.table(target).where(f"city = '{city.key}'").groupBy("price_class").count())

# COMMAND ----------

dbutils.notebook.exit(f'{{"city": "{city.key}", "silver_rows": {report["rows_out"]}}}')
