from __future__ import annotations

import json
from datetime import datetime, timezone

from pyspark.sql import SparkSession
from pyspark.sql import types as T

from airbnb.config import Lakehouse

SCHEMA = T.StructType([
    T.StructField("run_at", T.TimestampType()),
    T.StructField("city", T.StringType()),
    T.StructField("snapshot_date", T.StringType()),
    T.StructField("step", T.StringType()),
    T.StructField("rows_in", T.LongType()),
    T.StructField("rows_out", T.LongType()),
    T.StructField("details", T.StringType()),
])


def log_step(spark: SparkSession, lh: Lakehouse, city: str, snapshot_date: str, step: str,
             rows_in: int | None, rows_out: int | None, details: dict | None = None) -> None:
    row = (datetime.now(timezone.utc).replace(tzinfo=None), city, snapshot_date, step,
           rows_in, rows_out, json.dumps(details or {}, default=str))
    (spark.createDataFrame([row], SCHEMA).write.format("delta").mode("append")
     .saveAsTable(lh.table("gold", "pipeline_runs")))
