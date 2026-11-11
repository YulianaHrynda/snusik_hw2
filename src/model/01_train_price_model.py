# Databricks notebook source
# MAGIC %md
# MAGIC # Price model — train on the training cities
# MAGIC 1. Load `airbnb_gold.listing_features` for the training cities. The 80/20 train/test split
# MAGIC    was fixed in gold (hash of the listing id) before any model existed.
# MAGIC 2. **Baseline:** median training price for the same room type and `accommodates`.
# MAGIC 3. **Model choice on the training split only:** 5-fold cross-validated MAE of a few candidates.
# MAGIC 4. Refit the chosen model on the whole training split; evaluate **once** on the test split.
# MAGIC 5. Write predictions for every listing to `airbnb_gold.price_predictions`, metrics to
# MAGIC    `airbnb_gold.model_metrics`, save the model as version `model_version`.

# COMMAND ----------

import json
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

from airbnb.tables import pandas_to_spark
from airbnb.config import is_databricks, lakehouse_from, parse_city_list, read_params
from airbnb.modelling import (FEATURE_SETS, RANDOM_STATE, MedianBaseline, add_comps, metrics_rows,
                              prediction_frame, prepare, save_artifact)

# feature_set: "v1" = the original week-1 model, "comps" = v2 (after city C, see docs/city_c_report.md)
params = read_params(dbutils, train_cities='["boston", "jersey-city"]', model_version="v2", feature_set="comps",
                     model_name="auto", overwrite="false", catalog="workspace", raw_root="", artifacts_root="")
lh = lakehouse_from(params)
train_cities = parse_city_list(params["train_cities"])
version = params["model_version"]
fs = FEATURE_SETS[params["feature_set"]]
FEATURES, CANDIDATES = fs["features"], fs["candidates"]
print("training cities:", train_cities, "| model version:", version, "| feature set:", params["feature_set"])

# COMMAND ----------

city_list = ", ".join(f"'{c}'" for c in train_cities)
pdf = spark.table(lh.table("gold", "listing_features")).where(f"city IN ({city_list})").toPandas()
missing_cities = set(train_cities) - set(pdf["city"])
if missing_cities:
    raise ValueError(f"no gold features for {missing_cities}; run the city pipeline first")

data = prepare(pdf)
if fs["comps"]:
    data = add_comps(data)  # comps from each city's own training split, never the listing itself or its host
trainable = data[data["is_trainable"].astype(bool)]
train = trainable[trainable["split"] == "train"].reset_index(drop=True)
test = trainable[trainable["split"] == "test"].reset_index(drop=True)
print(pd.crosstab(trainable["city"], trainable["split"]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model choice: 5-fold CV on the training split (the test split is not touched here)

# COMMAND ----------

kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv = {name: [] for name in ["baseline", *CANDIDATES]}
for fold, (i_tr, i_va) in enumerate(kf.split(train)):
    tr, va = train.iloc[i_tr], train.iloc[i_va]
    cv["baseline"].append(np.mean(np.abs(MedianBaseline().fit(tr, tr["price"]).predict(va) - va["price"])))
    for name, make in CANDIDATES.items():
        m = make().fit(tr[FEATURES], tr["price"])
        cv[name].append(np.mean(np.abs(m.predict(va[FEATURES]) - va["price"])))
cv_table = pd.DataFrame({k: {"cv_mae_mean": np.mean(v), "cv_mae_std": np.std(v)} for k, v in cv.items()}).T
cv_table = cv_table.sort_values("cv_mae_mean")
display(cv_table.reset_index().rename(columns={"index": "candidate"}))

model_name = params["model_name"]
if model_name == "auto":
    model_name = cv_table.drop(index="baseline").index[0]
print("chosen model:", model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fit on the full training split, evaluate once on the test split

# COMMAND ----------

baseline = MedianBaseline().fit(train, train["price"])
model = CANDIDATES[model_name]().fit(train[FEATURES], train["price"])

pred_all = model.predict(data[FEATURES])
base_all = baseline.predict(data)
city_role = {c: "training" for c in train_cities}
preds = prediction_frame(pdf, pred_all, base_all, version, city_role)

scored = preds.assign(price=preds["actual_price"])
rows = []
for split in ("test", "train"):
    part = scored[(scored["split"] == split) & scored["is_trainable"].astype(bool)]
    rows += metrics_rows(part, version, model_name, split, "trainable")
# robustness: the same test listings plus the implausible prices that the main metric excludes
priced_test = scored[(scored["split"] == "test") & scored["actual_price"].notna()]
rows += metrics_rows(priced_test, version, model_name, "test", "all_priced")
metrics = pd.DataFrame(rows)
for name, v in cv.items():
    metrics = pd.concat([metrics, pd.DataFrame([{"model_version": version, "model_name": name, "split": "cv",
                                                 "scope": "trainable", "city": "ALL", "n": len(train),
                                                 "model_mae": float(np.mean(v))}])], ignore_index=True)
metrics["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
display(metrics[metrics["split"] != "cv"][["split", "scope", "city", "n", "model_mae", "baseline_mae",
                                           "mae_improvement_pct", "model_median_ae", "model_mape", "model_bias"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Which features matter (permutation importance on the test split)

# COMMAND ----------

imp = permutation_importance(model, test[FEATURES], test["price"], scoring="neg_mean_absolute_error",
                             n_repeats=5, random_state=RANDOM_STATE)
importance = (pd.DataFrame({"feature": FEATURES, "mae_increase": imp.importances_mean, "std": imp.importances_std})
              .sort_values("mae_increase", ascending=False).reset_index(drop=True))
display(importance.head(20))

err_by_room = (scored[(scored["split"] == "test") & scored["is_trainable"].astype(bool)]
               .groupby(["city", "room_type"])
               .apply(lambda d: pd.Series({"n": len(d), "model_mae": d["abs_error"].mean(),
                                           "baseline_mae": (d["baseline_price"] - d["actual_price"]).abs().mean()}))
               .reset_index())
display(err_by_room)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist: model artifact, predictions, metrics

# COMMAND ----------

path = save_artifact(lh.artifacts_root, version, {
    "model": model, "baseline": baseline, "model_name": model_name, "features": FEATURES,
    "feature_set": params["feature_set"],
    "train_cities": train_cities, "cv": cv_table.reset_index().to_dict("records"),
    "test_metrics": [r for r in rows if r["split"] == "test"],
    "feature_importance": importance.to_dict("records"),
}, overwrite=params["overwrite"].lower() == "true")
print("saved", path)

pred_table = lh.table("gold", "price_predictions")
metric_table = lh.table("gold", "model_metrics")
where_cities = f"model_version = '{version}' AND city IN ({city_list})"
(pandas_to_spark(spark, preds).write.format("delta").mode("overwrite").option("replaceWhere", where_cities)
 .option("mergeSchema", "true").saveAsTable(pred_table))
(pandas_to_spark(spark, metrics.astype({"n": "int64"})).write.format("delta").mode("overwrite")
 .option("replaceWhere", f"model_version = '{version}' AND city IN ({city_list}, 'ALL')")
 .option("mergeSchema", "true").saveAsTable(metric_table))
print(f"{pred_table}: {len(preds)} rows | {metric_table}: {len(metrics)} rows")

# COMMAND ----------

if is_databricks():  # experiment tracking + Unity Catalog registry (best effort, the joblib above is the source of truth)
    try:
        import mlflow
        from mlflow.models import infer_signature

        user = spark.sql("SELECT current_user()").first()[0]
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(f"/Users/{user}/airbnb_price_model")
        with mlflow.start_run(run_name=f"{version}-{model_name}"):
            mlflow.log_params({"model_version": version, "model_name": model_name, "train_cities": ",".join(train_cities)})
            test_all = [r for r in rows if r["split"] == "test" and r["scope"] == "trainable"]
            mlflow.log_metrics({f"test_mae_{r['city']}": r["model_mae"] for r in test_all})
            mlflow.log_metrics({f"test_baseline_mae_{r['city']}": r["baseline_mae"] for r in test_all})
            sample = train[FEATURES].head(50)
            mlflow.sklearn.log_model(model, "model", input_example=sample.head(5),
                                     signature=infer_signature(sample, model.predict(sample)),
                                     registered_model_name=f"{lh.schema('gold')}.price_model")
    except Exception as exc:  # noqa: BLE001
        print("MLflow logging skipped:", repr(exc))

# COMMAND ----------

test_summary = {r["city"]: {"model_mae": round(r["model_mae"], 2), "baseline_mae": round(r["baseline_mae"], 2)}
                for r in rows if r["split"] == "test" and r["scope"] == "trainable"}
dbutils.notebook.exit(json.dumps({"model_version": version, "model_name": model_name, "test": test_summary}))
