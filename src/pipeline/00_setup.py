# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the medallion schemas and the volumes for raw files and model artifacts.
# MAGIC Idempotent — every job run starts with it.

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

from airbnb.config import is_databricks, lakehouse_from, read_params

params = read_params(dbutils, catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)

# COMMAND ----------

for layer in ("bronze", "silver", "gold"):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {lh.schema(layer)} COMMENT 'Airbnb lakehouse — {layer} layer'")

if is_databricks():
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {lh.schema('bronze')}.raw COMMENT 'Inside Airbnb files as downloaded'")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {lh.schema('gold')}.artifacts COMMENT 'Fitted price models'")
else:
    os.makedirs(lh.raw_root, exist_ok=True)
    os.makedirs(lh.artifacts_root, exist_ok=True)

print("schemas:", [lh.schema(x) for x in ("bronze", "silver", "gold")])
print("raw root:", lh.raw_root, "| artifacts root:", lh.artifacts_root)
