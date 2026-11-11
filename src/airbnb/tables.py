"""Delta write helpers shared by all layers."""
from __future__ import annotations

from pyspark.sql import DataFrame


def write_city_partition(df: DataFrame, table: str, city_key: str, partition: bool = True) -> None:
    """Replace one city's rows in a Delta table (creating the table on first write)."""
    writer = (
        df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"city = '{city_key}'")
        .option("mergeSchema", "true")
    )
    if partition:
        writer = writer.partitionBy("city")
    writer.saveAsTable(table)
