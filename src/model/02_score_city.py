# Databricks notebook source
# MAGIC %md
# MAGIC # Price model — score a city with an existing model version
# MAGIC Loads a saved model version **as is** and predicts the price of every listing of `city`
# MAGIC from `airbnb_gold.listing_features`. Used to onboard a new client city: nothing is refit.
# MAGIC
# MAGIC * For a city the model was not trained on, every listing is out-of-sample, so metrics are
# MAGIC   reported on all trainable listings (`split = 'all'`) and, for comparability with the
# MAGIC   training cities, on the same hash-based 20% test split (`split = 'test'`).
# MAGIC * The baseline is the one stored with the model: medians from the **training** cities.

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd

from airbnb.tables import pandas_to_spark
from airbnb.config import lakehouse_from, read_params
from airbnb.modelling import MedianBaseline, add_comps, drift_report, load_artifact, metrics_rows, prediction_frame, prepare
from airbnb.runlog import log_step

params = read_params(dbutils, city="boston", model_version="v1", catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)
city, version = params["city"], params["model_version"]

artifact = load_artifact(lh.artifacts_root, version)
model, baseline, features = artifact["model"], artifact["baseline"], artifact["features"]
uses_comps = artifact.get("feature_set", "v1") == "comps"
print(f"model {version}: {artifact['model_name']} trained on {artifact['train_cities']} at {artifact['created_at']}")

# COMMAND ----------

pdf = spark.table(lh.table("gold", "listing_features")).where(f"city = '{city}'").toPandas()
if pdf.empty:
    raise ValueError(f"no gold features for city '{city}'; run airbnb_city_pipeline first")
data = prepare(pdf)
if uses_comps:
    data = add_comps(data)

train_list = ", ".join(f"'{c}'" for c in artifact["train_cities"])
train_rows = prepare(spark.table(lh.table("gold", "listing_features"))
                     .where(f"city IN ({train_list}) AND split = 'train' AND is_trainable").toPandas())
if uses_comps:
    train_rows = add_comps(train_rows)
drift = drift_report(train_rows, data, features)
flagged = {k: v for k, v in drift.items() if max(v.values()) > 0.05}
print("input drift (> 5% of rows unseen / out of range):", json.dumps(flagged, indent=1))
log_step(spark, lh, city, "", f"score_drift_{version}", len(data), None, drift)

pred = model.predict(data[features])
base = baseline.predict(data)
city_role = {c: "training" for c in artifact["train_cities"]}
preds = prediction_frame(pdf, pred, base, version, city_role)
role = city_role.get(city, "new")
print(f"{city}: {len(preds)} listings scored as a {role} city")

# COMMAND ----------

scored = preds.assign(price=preds["actual_price"])
trainable = scored[scored["is_trainable"].astype(bool)]
rows = []
rows += metrics_rows(trainable[trainable["split"] == "test"], version, artifact["model_name"], "test", "trainable")
if role == "new":
    rows += metrics_rows(trainable, version, artifact["model_name"], "all", "trainable")
    rows += metrics_rows(scored[scored["actual_price"].notna()], version, artifact["model_name"], "all", "all_priced")
rows = [r for r in rows if r["city"] != "ALL"]
if role == "new":
    own = data[data["is_trainable"].astype(bool) & data["split"].eq("train")]
    local_base = pd.Series(MedianBaseline().fit(own, own["price"]).predict(data), index=preds.index)
    for r in rows:
        part = trainable[trainable["split"].eq(r["split"])] if r["split"] == "test" else (
            trainable if r["scope"] == "trainable" else scored[scored["actual_price"].notna()])
        r["baseline_local_mae"] = float((local_base.loc[part.index] - part["actual_price"]).abs().mean())
metrics = pd.DataFrame(rows)
metrics["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
display(metrics[["split", "scope", "city", "n", "model_mae", "baseline_mae", "mae_improvement_pct",
                 "model_median_ae", "model_mape", "model_bias"]])

# COMMAND ----------

pred_table = lh.table("gold", "price_predictions")
metric_table = lh.table("gold", "model_metrics")
where = f"model_version = '{version}' AND city = '{city}'"
(pandas_to_spark(spark, preds).write.format("delta").mode("overwrite").option("replaceWhere", where)
 .option("mergeSchema", "true").saveAsTable(pred_table))
(pandas_to_spark(spark, metrics.astype({"n": "int64"})).write.format("delta").mode("overwrite")
 .option("replaceWhere", where).option("mergeSchema", "true").saveAsTable(metric_table))
print(f"{pred_table}: {len(preds)} rows for {city} | {metric_table}: {len(metrics)} rows")

# COMMAND ----------

summary = {r["split"] + "/" + r["scope"]: {"n": r["n"], "model_mae": r["model_mae"], "baseline_mae": r["baseline_mae"]}
           for r in rows}
dbutils.notebook.exit(json.dumps({"city": city, "model_version": version, "metrics": summary}, default=str))
