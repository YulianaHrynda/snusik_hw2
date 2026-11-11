"""Silver: typed, cleaned, de-duplicated listings (rules from the EDA notebook, section 6)."""
from __future__ import annotations

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

# Outlier rule from the EDA (section 5): a price per guest at or above this is not a market price.
IMPLAUSIBLE_PRICE_PER_GUEST = 750.0
# Listings that cannot be booked for less than this many nights are monthly rentals (EDA 4.2).
MONTHLY_MIN_NIGHTS = 28


def _bool(col: str) -> str:
    return f"CASE WHEN `{col}` = 't' THEN true WHEN `{col}` = 'f' THEN false END AS {col}"


def _int(col: str, alias: str | None = None) -> str:
    # via DOUBLE: under ANSI mode try_cast('1.0' AS INT) is NULL, and some files write '1.0'
    return f"try_cast(try_cast(`{col}` AS DOUBLE) AS INT) AS {alias or col}"


def _double(col: str, alias: str | None = None) -> str:
    return f"try_cast(`{col}` AS DOUBLE) AS {alias or col}"


PRICE_EXPR = "try_cast(regexp_replace(price, '[$,]', '') AS DOUBLE)"
BATHROOMS_EXPR = (
    "CASE WHEN lower(bathrooms_text) LIKE '%half-bath%' THEN 0.5 "
    "ELSE try_cast(nullif(regexp_extract(bathrooms_text, '([0-9]+([.][0-9]+)?)', 1), '') AS DOUBLE) END"
)

REVIEW_SCORES = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value",
]

# Columns deliberately NOT carried into silver (EDA section 7): price_quote_* and
# estimated_revenue_l365d are computed from the price; the 14 always-empty columns; URLs/images.
SILVER_SELECT = [
    "try_cast(id AS BIGINT) AS listing_id",
    "city",
    "snapshot_date",
    "try_cast(last_scraped AS DATE) AS last_scraped",
    "coalesce(source = 'previous scrape', false) AS is_stale",
    "listing_url",
    "name",
    "try_cast(host_id AS BIGINT) AS host_id",
    "host_name",
    _bool("host_is_superhost"),
    _bool("host_identity_verified"),
    _bool("host_has_profile_pic"),
    _double("hosts_time_as_host_years", "host_years"),
    _int("calculated_host_listings_count", "host_listings_count"),
    "neighbourhood_cleansed AS neighbourhood",
    "nullif(trim(neighbourhood_group_cleansed), '') AS neighbourhood_group",  # e.g. NYC borough; empty in A/B
    _double("latitude"),
    _double("longitude"),
    "property_type",
    "room_type",
    _int("accommodates"),
    _double("bedrooms"),
    _double("beds"),
    "bathrooms_text",
    f"{BATHROOMS_EXPR} AS bathrooms",
    "coalesce(lower(bathrooms_text) LIKE '%shared%', false) AS bath_shared",
    "from_json(amenities, 'array<string>') AS amenities",
    f"{PRICE_EXPR} AS price",
    _int("minimum_nights"),
    _int("maximum_nights"),
    _bool("has_availability"),
    _int("availability_30"),
    _int("availability_60"),
    _int("availability_90"),
    _int("availability_365"),
    _int("number_of_reviews"),
    _int("number_of_reviews_ltm"),
    "try_cast(first_review AS DATE) AS first_review",
    "try_cast(last_review AS DATE) AS last_review",
    *[_double(c) for c in REVIEW_SCORES],
    _double("reviews_per_month"),
    "license",
]

# Source columns the pipeline cannot work without; any other column it reads is optional and is
# added as NULL (with a warning) when a city's file does not have it.
REQUIRED_COLUMNS = ["id", "price", "room_type", "accommodates", "latitude", "longitude", "neighbourhood_cleansed"]

# Expected ranges, from Boston and Jersey City. Outside them the run continues but logs a warning.
PROFILE_LIMITS = {
    "missing_price_share": 0.25,
    "implausible_share": 0.002,
    "monthly_stay_share": 0.60,
    "stale_share": 0.25,
}

# raw column -> silver column, checked for values that were present but failed to parse
PARSE_CHECKS = {
    "id": "listing_id", "price": "price", "accommodates": "accommodates", "bedrooms": "bedrooms",
    "beds": "beds", "latitude": "latitude", "longitude": "longitude",
    "minimum_nights": "minimum_nights", "availability_365": "availability_365",
    "number_of_reviews": "number_of_reviews", "review_scores_rating": "review_scores_rating",
    "amenities": "amenities",
}


# Every raw column SILVER_SELECT reads.
SOURCE_COLUMNS = [
    "id", "last_scraped", "source", "listing_url", "name", "host_id", "host_name", "host_is_superhost",
    "host_identity_verified", "host_has_profile_pic", "hosts_time_as_host_years", "calculated_host_listings_count",
    "neighbourhood_cleansed", "neighbourhood_group_cleansed", "latitude", "longitude", "property_type", "room_type",
    "accommodates", "bedrooms", "beds", "bathrooms_text", "amenities", "price", "minimum_nights", "maximum_nights",
    "has_availability", "availability_30", "availability_60", "availability_90", "availability_365",
    "number_of_reviews", "number_of_reviews_ltm", "first_review", "last_review", *REVIEW_SCORES,
    "reviews_per_month", "license",
]


def align_schema(bronze: DataFrame) -> tuple[DataFrame, list[str]]:
    """Fail on missing required columns; add missing optional ones as NULL and report them."""
    present = set(bronze.columns)
    missing_required = [c for c in REQUIRED_COLUMNS if c not in present]
    if missing_required:
        raise ValueError(f"source file lacks required columns {missing_required}")
    missing_optional = sorted(c for c in SOURCE_COLUMNS if c not in present)
    for c in missing_optional:
        bronze = bronze.withColumn(c, F.lit(None).cast("string"))
    return bronze, missing_optional


def type_and_clean(bronze: DataFrame) -> DataFrame:
    return bronze.selectExpr(*SILVER_SELECT)


def profile(silver: DataFrame) -> tuple[dict, list[str]]:
    """Shares that describe the market and the target; warnings when outside PROFILE_LIMITS."""
    row = silver.agg(
        F.avg(F.expr("CAST(price_class = 'missing' AS DOUBLE)")).alias("missing_price_share"),
        F.avg(F.expr("CAST(price_class = 'implausible' AS DOUBLE)")).alias("implausible_share"),
        F.avg(F.col("monthly_stay").cast("double")).alias("monthly_stay_share"),
        F.avg(F.col("is_stale").cast("double")).alias("stale_share"),
        F.avg(F.expr("CAST(neighbourhood_group IS NOT NULL AS DOUBLE)")).alias("neighbourhood_group_filled_share"),
    ).first().asDict()
    prof = {k: round(float(v or 0), 4) for k, v in row.items()}
    warnings = [f"{k} = {prof[k]:.1%} is above the expected {limit:.1%}"
                for k, limit in PROFILE_LIMITS.items() if prof[k] > limit]
    return prof, warnings


def parse_report(bronze: DataFrame, typed: DataFrame) -> dict:
    """Count values that were non-null in bronze but null after typing (silent parse failures)."""
    raw_counts = bronze.agg(*[F.count(F.col(f"`{r}`")).alias(r) for r in PARSE_CHECKS]).first().asDict()
    typed_counts = typed.agg(*[F.count(F.col(s)).alias(s) for s in PARSE_CHECKS.values()]).first().asDict()
    return {r: int(raw_counts[r] - typed_counts[s]) for r, s in PARSE_CHECKS.items()}


def deduplicate(df: DataFrame) -> DataFrame:
    """One row per listing: keep the most recently scraped, non-stale record."""
    w = Window.partitionBy("listing_id").orderBy(F.col("last_scraped").desc_nulls_last(), F.col("is_stale").asc())
    return df.withColumn("_rn", F.row_number().over(w)).where("_rn = 1").drop("_rn")


def add_flags(df: DataFrame) -> DataFrame:
    """Derived flags and the outlier class (EDA sections 3.2, 4.2 and 5)."""
    p99 = df.selectExpr("percentile(price, 0.99) AS p99").first()["p99"]  # exact, NULL if no prices
    extreme = f"WHEN price > {p99} THEN 'extreme' " if p99 is not None else ""
    price_class = (
        "CASE WHEN price IS NULL THEN 'missing' "
        f"WHEN try_divide(price, accommodates) >= {IMPLAUSIBLE_PRICE_PER_GUEST} THEN 'implausible' "
        f"{extreme}ELSE 'normal' END"
    )
    return (
        df.withColumn("price_class", F.expr(price_class))
        .withColumn("has_valid_price", F.expr("price_class IN ('normal', 'extreme')"))
        .withColumn("monthly_stay", F.expr(f"coalesce(minimum_nights >= {MONTHLY_MIN_NIGHTS}, false)"))
        .withColumn("has_reviews", F.expr("coalesce(number_of_reviews > 0, false)"))
        .withColumn("calendar_blocked", F.expr("availability_365 = 0"))
        .withColumn("_built_at", F.current_timestamp())
    )


def build_silver(bronze: DataFrame) -> tuple[DataFrame, dict]:
    bronze, missing_optional = align_schema(bronze)
    typed = type_and_clean(bronze)
    report = {"rows_in": bronze.count(), "parse_failures": parse_report(bronze, typed),
              "missing_optional_columns": missing_optional}
    valid = typed.where("listing_id IS NOT NULL")
    report["dropped_null_id"] = report["rows_in"] - valid.count()
    deduped = deduplicate(valid)
    silver = add_flags(deduped)
    report["rows_out"] = silver.count()
    report["dropped_duplicates"] = report["rows_in"] - report["dropped_null_id"] - report["rows_out"]
    report["profile"], report["warnings"] = profile(silver)
    report["warnings"] += [f"source column '{c}' missing, filled with NULL" for c in missing_optional]
    return silver, report
