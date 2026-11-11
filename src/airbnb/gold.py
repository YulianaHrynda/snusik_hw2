"""Gold: consumption tables built from silver for one city."""
from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# Share of listings per city that goes to the test set. Decided before any model was trained.
TEST_PERCENT = 20
SPLIT_EXPR = f"CASE WHEN pmod(xxhash64(CAST(listing_id AS STRING)), 100) < {TEST_PERCENT} THEN 'test' ELSE 'train' END"

# Amenity flags used as features (case-insensitive regex on each amenity string)
AMENITY_FLAGS = {
    "has_air_conditioning": r"air conditioning|\bac\b",
    "has_pool": r"^(private |shared )?(indoor |outdoor )?pool( -|$)",
    "has_hot_tub": r"hot tub",
    "has_gym": r"\bgym\b",
    "has_elevator": r"^elevator$",
    "has_free_parking": r"^free .*parking on premises",
    "has_washer": r"^(free |paid )?washer",
    "has_dishwasher": r"^dishwasher$",
    "has_workspace": r"workspace",
    "has_self_checkin": r"^self check-in$",
    "has_outdoor_space": r"patio|balcony|backyard",
    "has_skyline_or_water_view": r"skyline view|water view|ocean view|harbor view|river view|bay view|waterfront",
    "pets_allowed": r"^pets allowed$",
}


def _amenity_expr(pattern: str) -> str:
    escaped = pattern.replace("\\", "\\\\")
    return f"coalesce(exists(amenities, a -> lower(a) RLIKE '{escaped}'), false)"


def build_listings(silver: DataFrame, city_name: str) -> DataFrame:
    """Curated listing-level table for the dashboard and Genie (no free text, no leak columns)."""
    return silver.select(
        "listing_id", "city", F.lit(city_name).alias("city_name"), "snapshot_date", "name", "listing_url",
        "neighbourhood_group", "neighbourhood", "latitude", "longitude", "room_type", "property_type",
        "accommodates", "bedrooms", "beds", "bathrooms", "bath_shared",
        "price", "price_class", "has_valid_price",
        F.expr("try_divide(price, accommodates)").alias("price_per_guest"),
        "minimum_nights", "monthly_stay",
        "availability_30", "availability_90", "availability_365", "calendar_blocked",
        "number_of_reviews", "number_of_reviews_ltm", "has_reviews", "review_scores_rating",
        "review_scores_cleanliness", "review_scores_location", "review_scores_value",
        "host_id", "host_name", "host_is_superhost", "host_listings_count", "is_stale", "last_scraped",
    )


def build_city_metrics(silver: DataFrame, city_name: str) -> DataFrame:
    vp = "has_valid_price"
    return silver.groupBy("city", "snapshot_date").agg(
        F.lit(city_name).alias("city_name"),
        F.count("*").alias("listings"),
        F.sum(F.col(vp).cast("int")).alias("priced_listings"),
        F.countDistinct("host_id").alias("hosts"),
        F.countDistinct("neighbourhood").alias("neighbourhoods"),
        F.countDistinct("neighbourhood_group").alias("neighbourhood_groups"),
        F.expr(f"percentile(CASE WHEN {vp} THEN price END, 0.5)").alias("median_price"),
        F.expr(f"percentile(CASE WHEN {vp} THEN price END, 0.25)").alias("p25_price"),
        F.expr(f"percentile(CASE WHEN {vp} THEN price END, 0.75)").alias("p75_price"),
        F.expr(f"percentile(CASE WHEN {vp} AND room_type = 'Entire home/apt' THEN price END, 0.5)").alias("median_price_entire_home"),
        F.expr(f"percentile(CASE WHEN {vp} AND room_type = 'Private room' THEN price END, 0.5)").alias("median_price_private_room"),
        F.expr(f"percentile(CASE WHEN {vp} AND NOT monthly_stay THEN price END, 0.5)").alias("median_price_short_stay"),
        F.expr(f"percentile(CASE WHEN {vp} AND monthly_stay THEN price END, 0.5)").alias("median_price_monthly_stay"),
        F.avg(F.expr("CAST(room_type = 'Entire home/apt' AS INT)")).alias("entire_home_share"),
        F.avg(F.expr("CAST(room_type = 'Private room' AS INT)")).alias("private_room_share"),
        F.avg(F.expr("CAST(room_type = 'Hotel room' AS INT)")).alias("hotel_room_share"),
        F.avg(F.expr("CAST(room_type = 'Shared room' AS INT)")).alias("shared_room_share"),
        F.avg(F.col("monthly_stay").cast("int")).alias("monthly_stay_share"),
        F.expr("percentile(availability_30, 0.5)").alias("median_availability_30"),
        F.expr("percentile(availability_365, 0.5)").alias("median_availability_365"),
        F.avg(F.col("calendar_blocked").cast("int")).alias("blocked_calendar_share"),
        F.avg("review_scores_rating").alias("avg_review_score"),
        F.avg(F.col("has_reviews").cast("int")).alias("reviewed_share"),
        F.avg(F.col("host_is_superhost").cast("int")).alias("superhost_share"),
        F.avg(F.expr("CAST(host_listings_count > 1 AS INT)")).alias("multi_listing_host_share"),
        F.avg(F.col("is_stale").cast("int")).alias("stale_share"),
        F.current_timestamp().alias("_built_at"),
    )


def build_neighbourhood_metrics(silver: DataFrame, city_name: str) -> DataFrame:
    vp = "has_valid_price"
    agg = silver.groupBy("city", "neighbourhood_group", "neighbourhood").agg(
        F.lit(city_name).alias("city_name"),
        F.count("*").alias("listings"),
        F.sum(F.col(vp).cast("int")).alias("priced_listings"),
        F.expr(f"percentile(CASE WHEN {vp} THEN price END, 0.5)").alias("median_price"),
        F.expr(f"percentile(CASE WHEN {vp} AND room_type = 'Entire home/apt' THEN price END, 0.5)").alias("median_price_entire_home"),
        F.avg(F.expr("CAST(room_type = 'Entire home/apt' AS INT)")).alias("entire_home_share"),
        F.avg("review_scores_rating").alias("avg_review_score"),
        F.expr("percentile(availability_365, 0.5)").alias("median_availability_365"),
        F.avg("latitude").alias("latitude"),
        F.avg("longitude").alias("longitude"),
    )
    return agg.withColumn(
        "price_rank_in_city", F.expr("dense_rank() OVER (PARTITION BY city ORDER BY median_price DESC NULLS LAST)")
    ).withColumn("_built_at", F.current_timestamp())


def build_listing_features(silver: DataFrame) -> DataFrame:
    """Model-ready features, the label and the fixed train/test split."""
    amenity_cols = [F.expr(_amenity_expr(p)).alias(name) for name, p in AMENITY_FLAGS.items()]
    return silver.select(
        "listing_id", "city", "snapshot_date",
        F.expr(SPLIT_EXPR).alias("split"),
        "price", "price_class",
        F.col("has_valid_price").alias("is_trainable"),
        "room_type", "property_type", "neighbourhood_group", "neighbourhood", "latitude", "longitude",
        "accommodates", "bedrooms", "beds", "bathrooms", "bath_shared",
        F.expr("CASE WHEN amenities IS NULL THEN NULL ELSE size(amenities) END").alias("n_amenities"),
        *amenity_cols,
        "minimum_nights", "monthly_stay",
        "host_is_superhost", "host_identity_verified", "host_listings_count", "host_years",
        "number_of_reviews", "has_reviews", "review_scores_rating", "review_scores_cleanliness",
        "review_scores_location", "review_scores_value",
        F.current_timestamp().alias("_built_at"),
    )
