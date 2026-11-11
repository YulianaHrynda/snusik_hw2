"""Bronze: land the raw file and load it as-is (every column a string) with lineage columns."""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from airbnb.config import City, Lakehouse

GZIP_MAGIC = b"\x1f\x8b"


def landing_file(lh: Lakehouse, city: City) -> str:
    return os.path.join(lh.raw_root, city.key, city.snapshot_date, "listings.csv.gz")


def ensure_raw_file(lh: Lakehouse, city: City) -> tuple[str, bool]:
    """Download the snapshot into the landing zone unless it is already there.

    Returns (path, downloaded_now). The file is first written to local temp storage and then
    copied, so a failed download never leaves a truncated file in the landing zone.
    """
    target = landing_file(lh, city)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target, False

    os.makedirs(os.path.dirname(target), exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".csv.gz")
    os.close(fd)
    try:
        with urllib.request.urlopen(city.url, timeout=300) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        with open(tmp, "rb") as f:
            if f.read(2) != GZIP_MAGIC:
                raise ValueError("downloaded file is not gzip")
        shutil.copyfile(tmp, target)
    except Exception as exc:  # noqa: BLE001 - re-raised with an actionable message
        raise RuntimeError(
            f"Could not download {city.url} ({exc!r}). If this compute has no internet access, "
            f"upload the file with: scripts/upload_raw.sh {city.key}"
        ) from exc
    finally:
        os.remove(tmp)
    return target, True


def read_listings_csv(spark: SparkSession, path: str) -> DataFrame:
    """Read an Inside Airbnb listings file without typing anything.

    multiLine + '"' escaping are required: descriptions contain line breaks and some cells hold
    JSON with doubled quotes.
    """
    return (
        spark.read.option("header", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(path)
    )


def to_bronze(raw: DataFrame, city: City, source_file: str) -> DataFrame:
    return (
        raw.withColumn("city", F.lit(city.key))
        .withColumn("snapshot_date", F.lit(city.snapshot_date))
        .withColumn("_source_file", F.lit(source_file))
        .withColumn("_ingested_at", F.current_timestamp())
    )
