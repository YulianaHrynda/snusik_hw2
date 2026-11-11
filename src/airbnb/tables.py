from __future__ import annotations

from pyspark.sql import DataFrame


def write_city_partition(df: DataFrame, table: str, city_key: str, partition: bool = True) -> None:
    writer = (
        df.write.format("delta")
        .mode("overwrite")
        .option("replaceWhere", f"city = '{city_key}'")
        .option("mergeSchema", "true")
    )
    if partition:
        writer = writer.partitionBy("city")
    writer.saveAsTable(table)


def pandas_to_spark(spark, pdf):
    from pyspark.sql import functions as F

    sdf = spark.createDataFrame(pdf)
    for name, dtype in sdf.dtypes:
        if dtype in ("double", "float"):
            sdf = sdf.withColumn(name, F.when(F.isnan(name), None).otherwise(F.col(name)))
    return sdf
