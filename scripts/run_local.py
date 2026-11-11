#!/usr/bin/env python
"""Run this repo's Databricks notebooks on a local Spark 4 + Delta session.

The notebooks are executed unchanged: a tiny shim provides `spark`, `dbutils.widgets`,
`dbutils.notebook.exit` and `display`. Tables go to the local metastore under catalog
`spark_catalog`, the landing zone is ./data/raw and model artifacts go to
./.local_lakehouse/artifacts. The presets mirror the jobs in resources/*.yml.

    python scripts/run_local.py city-pipeline --city boston
    python scripts/run_local.py train --train-cities boston,jersey-city --model-version v1
    python scripts/run_local.py onboard --city new-york-city --model-version v1
    python scripts/run_local.py notebook src/pipeline/02_build_silver.py --param city=boston
    python scripts/run_local.py sql "SELECT * FROM spark_catalog.airbnb_gold.city_metrics"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOCAL = os.path.join(REPO, ".local_lakehouse")
sys.path.insert(0, os.path.join(REPO, "src"))

DEFAULT_PARAMS = {
    "catalog": "spark_catalog",
    "raw_root": os.path.join(REPO, "data", "raw"),
    "artifacts_root": os.path.join(LOCAL, "artifacts"),
}

CITY_PIPELINE = [
    "src/pipeline/00_setup.py",
    "src/pipeline/01_ingest_bronze.py",
    "src/pipeline/02_build_silver.py",
    "src/pipeline/03_build_gold.py",
]


class NotebookExit(Exception):
    def __init__(self, value):
        super().__init__(value)
        self.value = value


class _Widgets:
    def __init__(self, params: dict):
        self._params = params
        self._values: dict = {}

    def text(self, name, default="", label=None):
        self._values[name] = self._params.get(name, default)

    dropdown = combobox = lambda self, name, default="", choices=None, label=None: self.text(name, default)

    def get(self, name):
        if name in self._values:
            return self._values[name]
        return self._params[name]

    def removeAll(self):
        self._values.clear()


class _Notebook:
    @staticmethod
    def exit(value):
        raise NotebookExit(value)


class DBUtils:
    def __init__(self, params: dict):
        self.widgets = _Widgets(params)
        self.notebook = _Notebook()


def display(obj, *args, **kwargs):
    if hasattr(obj, "show"):
        obj.show(20, truncate=60)
    else:
        print(obj)


def spark_session():
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17")
    os.environ["PYSPARK_PYTHON"] = sys.executable  # workers must use the driver's Python
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    os.makedirs(LOCAL, exist_ok=True)
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("airbnb-local")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.sql.warehouse.dir", os.path.join(LOCAL, "warehouse"))
        .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                f"jdbc:derby:;databaseName={os.path.join(LOCAL, 'metastore_db')};create=true")
        .config("spark.driver.extraJavaOptions", f"-Dderby.system.home={LOCAL}")
        .config("spark.sql.ansi.enabled", "true")  # Databricks serverless runs in ANSI mode
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.ui.showConsoleProgress", "false")
        .enableHiveSupport()
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def run_notebook(spark, path: str, params: dict):
    """Execute a Databricks-source notebook (.py) with the notebook's folder as working dir."""
    path = os.path.join(REPO, path) if not os.path.isabs(path) else path
    with open(path, encoding="utf-8") as f:
        source = f.read()
    code = "\n".join(line for line in source.splitlines() if not line.startswith("# MAGIC"))
    scope = {"__name__": "__main__", "__file__": path, "spark": spark,
             "dbutils": DBUtils({**DEFAULT_PARAMS, **params}), "display": display}
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    started = time.time()
    print(f"\n=== {os.path.relpath(path, REPO)} {params}")
    try:
        exec(compile(code, path, "exec"), scope)
        result = None
    except NotebookExit as e:
        result = e.value
    finally:
        os.chdir(cwd)
    print(f"=== done in {time.time() - started:.0f}s" + (f" -> {result}" if result else ""))
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("city-pipeline", help="setup -> bronze -> silver -> gold for one or more cities")
    p.add_argument("--city", required=True, help="city key or comma-separated keys")

    p = sub.add_parser("train", help="mirror of job airbnb_train_price_model")
    p.add_argument("--train-cities", default="boston,jersey-city")
    p.add_argument("--model-version", default="v2")
    p.add_argument("--feature-set", default="comps", help='"v1" reproduces the week-1 model')
    p.add_argument("--model-name", default="auto")
    p.add_argument("--overwrite", default="false")
    p.add_argument("--skip-etl", action="store_true")

    p = sub.add_parser("onboard", help="mirror of job airbnb_onboard_city")
    p.add_argument("--city", required=True)
    p.add_argument("--model-version", default="v2")

    p = sub.add_parser("notebook", help="run a single notebook")
    p.add_argument("path")
    p.add_argument("--param", action="append", default=[], help="key=value widget value")

    p = sub.add_parser("sql", help="run a SQL query and print the result")
    p.add_argument("query")
    p.add_argument("--rows", type=int, default=50)

    args = ap.parse_args(argv)
    spark = spark_session()

    if args.cmd == "city-pipeline":
        for city in [c.strip() for c in args.city.split(",") if c.strip()]:
            for nb in CITY_PIPELINE:
                run_notebook(spark, nb, {"city": city})
    elif args.cmd == "train":
        cities = [c.strip() for c in args.train_cities.split(",") if c.strip()]
        if not args.skip_etl:
            for city in cities:
                for nb in CITY_PIPELINE:
                    run_notebook(spark, nb, {"city": city})
        run_notebook(spark, "src/model/01_train_price_model.py",
                     {"train_cities": ",".join(cities), "model_version": args.model_version, "feature_set": args.feature_set,
                      "model_name": args.model_name, "overwrite": args.overwrite})
    elif args.cmd == "onboard":
        for nb in CITY_PIPELINE:
            run_notebook(spark, nb, {"city": args.city})
        run_notebook(spark, "src/model/02_score_city.py", {"city": args.city, "model_version": args.model_version})
    elif args.cmd == "notebook":
        run_notebook(spark, args.path, dict(kv.split("=", 1) for kv in args.param))
    elif args.cmd == "sql":
        spark.sql(args.query).show(args.rows, truncate=False)


if __name__ == "__main__":
    main()
