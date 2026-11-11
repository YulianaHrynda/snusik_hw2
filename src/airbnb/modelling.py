"""Price model: features, baseline, candidate models, metrics and artifact storage."""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import BallTree
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from airbnb.gold import AMENITY_FLAGS

CATEGORICAL = ["city", "room_type", "property_type", "neighbourhood"]
NUMERIC = [
    "latitude", "longitude", "accommodates", "bedrooms", "beds", "bathrooms", "n_amenities",
    "minimum_nights", "host_listings_count", "host_years", "number_of_reviews",
    "review_scores_rating", "review_scores_cleanliness", "review_scores_location", "review_scores_value",
]
BOOLEAN = ["bath_shared", "monthly_stay", "host_is_superhost", "host_identity_verified", "has_reviews",
           *AMENITY_FLAGS]
FEATURES = CATEGORICAL + NUMERIC + BOOLEAN
TARGET = "price"
RANDOM_STATE = 42


# ----------------------------------------------------------------------------- data
def prepare(pdf: pd.DataFrame) -> pd.DataFrame:
    """Give every feature a model-friendly dtype (Spark booleans with NULLs arrive as objects)."""
    out = pdf.copy()
    for c in BOOLEAN:
        out[c] = out[c].map({True: 1.0, False: 0.0}).astype("float64")
    for c in NUMERIC:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    for c in CATEGORICAL:
        out[c] = out[c].astype("object").where(out[c].notna(), "unknown").astype(str)
    return out


# ----------------------------------------------------------------------------- baseline
class MedianBaseline:
    """Median training price for the same (room_type, accommodates).

    Unseen combinations fall back to the room-type median, then to the global median.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MedianBaseline":
        d = X[["room_type", "accommodates"]].assign(y=y.values)
        self.by_room_acc_ = d.groupby(["room_type", "accommodates"])["y"].median().to_dict()
        self.by_room_ = d.groupby("room_type")["y"].median().to_dict()
        self.global_ = float(d["y"].median())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = zip(X["room_type"], X["accommodates"])
        return np.array([
            self.by_room_acc_.get((r, a), self.by_room_.get(r, self.global_)) for r, a in keys
        ], dtype=float)

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(r, a, p) for (r, a), p in self.by_room_acc_.items()],
            columns=["room_type", "accommodates", "baseline_price"],
        )


# ----------------------------------------------------------------------------- models
def _onehot() -> OneHotEncoder:
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)


def make_ridge(alpha: float = 3.0):
    pre = ColumnTransformer([
        ("cat", _onehot(), CATEGORICAL),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         NUMERIC + BOOLEAN),
    ])
    reg = Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])
    return TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)


def make_hgb(loss: str = "squared_error", max_iter: int = 400, max_leaf_nodes: int = 31,
             learning_rate: float = 0.05, min_samples_leaf: int = 20):
    """Gradient-boosted trees on log price. NaNs are handled natively (no imputation)."""
    pre = ColumnTransformer([("cat", _onehot(), CATEGORICAL), ("num", "passthrough", NUMERIC + BOOLEAN)])
    hgb = HistGradientBoostingRegressor(
        loss=loss, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf, l2_regularization=1.0, early_stopping=False,
        random_state=RANDOM_STATE,
    )
    reg = Pipeline([("pre", pre), ("hgb", hgb)])
    return TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)


CANDIDATES = {
    "ridge_log": lambda: make_ridge(alpha=3.0),
    "hgb_log_l2": lambda: make_hgb(loss="squared_error"),
    "hgb_log_l1": lambda: make_hgb(loss="absolute_error"),
    "hgb_log_l1_deep": lambda: make_hgb(loss="absolute_error", max_iter=800, max_leaf_nodes=63, learning_rate=0.03),
}


# ----------------------------------------------------------------------------- v2: market comps
# Model change after city C (docs/city_c_report.md). v1 learned Boston/Jersey City location and
# price *levels* (city, neighbourhood names, raw coordinates); none of that transfers. v2 drops
# those features and anchors every prediction on comparable listings of the *same* city:
#   prediction = comp_price_seg * exp(model(features))
# so the price level comes from the local market and the model only learns relative adjustments.
# Comps come from the city's training-split listings with a valid price, never from the listing
# itself or from other listings of the same host.
COMP_K_SEGMENT = 20     # nearest listings of the same room type and stay type
COMP_K_LOCAL = 50       # nearest listings of any type (local market level)
EARTH_RADIUS_KM = 6371.0088
ANCHOR = "comp_price_seg"
COMP_FEATURES = ["comp_log_price_seg", "comp_log_price_local", "comp_accommodates_seg", "comp_radius_km_seg",
                 "rel_accommodates"]
CATEGORICAL_V2 = ["room_type", "property_type"]
NUMERIC_V2 = [c for c in NUMERIC if c not in ("latitude", "longitude")] + COMP_FEATURES
FEATURES_V2 = CATEGORICAL_V2 + NUMERIC_V2 + BOOLEAN


def segment(df: pd.DataFrame) -> pd.Series:
    room = df["room_type"].map({"Entire home/apt": "entire", "Private room": "private"}).fillna("other")
    stay = np.where(pd.to_numeric(df["monthly_stay"], errors="coerce").fillna(0) > 0.5, "monthly", "short")
    return room + "_" + stay


def _knn_comps(rows: pd.DataFrame, ref: pd.DataFrame, k: int, extra: int = 100) -> dict:
    n = len(rows)
    out = {"price": np.full(n, np.nan), "acc": np.full(n, np.nan), "radius": np.full(n, np.nan)}
    has_xy = rows[["latitude", "longitude"]].notna().all(axis=1).to_numpy()
    if len(ref) == 0 or not has_xy.any():
        return out
    tree = BallTree(np.radians(ref[["latitude", "longitude"]].to_numpy(dtype=float)), metric="haversine")
    dist, idx = tree.query(np.radians(rows.loc[has_xy, ["latitude", "longitude"]].to_numpy(dtype=float)),
                           k=min(len(ref), k + extra))
    ok = ((ref["listing_id"].to_numpy()[idx] != rows["listing_id"].to_numpy()[has_xy][:, None])
          & (ref["host_id"].to_numpy()[idx] != rows["host_id"].to_numpy()[has_xy][:, None]))
    keep = ok & (np.cumsum(ok, axis=1) <= k)
    with warnings.catch_warnings():  # rows without any usable comp stay NaN and get a fallback
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out["price"][has_xy] = np.nanmedian(np.where(keep, ref["price"].to_numpy(dtype=float)[idx], np.nan), axis=1)
        out["acc"][has_xy] = np.nanmedian(np.where(keep, ref["accommodates"].to_numpy(dtype=float)[idx], np.nan), axis=1)
        out["radius"][has_xy] = np.nanmax(np.where(keep, dist * EARTH_RADIUS_KM, np.nan), axis=1)
    return out


def add_comps(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-comp features per city (expects prepared data with split/is_trainable/host_id)."""
    out = df.copy()
    out["segment"] = segment(out)
    for c in (ANCHOR, "comp_price_local", "comp_accommodates_seg", "comp_radius_km_seg"):
        out[c] = np.nan
    is_ref = (out["split"].eq("train") & out["is_trainable"].astype(bool) & out["price"].notna()
              & out[["latitude", "longitude"]].notna().all(axis=1))
    for city, d in out.groupby("city"):
        ref_city = d[is_ref.loc[d.index]]
        if ref_city.empty:
            raise ValueError(f"city '{city}' has no priced training-split listings to use as comps")
        local = _knn_comps(d, ref_city, COMP_K_LOCAL)
        out.loc[d.index, "comp_price_local"] = local["price"]
        for seg, ds in d.groupby("segment"):
            ref = ref_city[ref_city["segment"] == seg]
            r = _knn_comps(ds, ref if len(ref) >= COMP_K_SEGMENT else ref_city, COMP_K_SEGMENT)
            out.loc[ds.index, ANCHOR] = r["price"]
            out.loc[ds.index, "comp_accommodates_seg"] = r["acc"]
            out.loc[ds.index, "comp_radius_km_seg"] = r["radius"]
        # fallbacks for rows without a usable comp: segment median, then city median
        rows = out.index.isin(d.index)
        seg_median = ref_city.groupby("segment")["price"].median()
        city_median = ref_city["price"].median()
        miss = rows & out[ANCHOR].isna()
        if miss.any():
            out.loc[miss, ANCHOR] = out.loc[miss, "segment"].map(seg_median).astype(float).fillna(city_median)
        miss = rows & out["comp_price_local"].isna()
        if miss.any():
            out.loc[miss, "comp_price_local"] = float(city_median)
    out["comp_log_price_seg"] = np.log(out[ANCHOR])
    out["comp_log_price_local"] = np.log(out["comp_price_local"])
    out["rel_accommodates"] = out["accommodates"] / out["comp_accommodates_seg"]
    return out


class CompsResidualModel(BaseEstimator, RegressorMixin):
    """Fits log(price) - log(anchor); predicts anchor * exp(prediction)."""

    def __init__(self, inner=None, anchor: str = ANCHOR):
        self.inner = inner
        self.anchor = anchor

    def fit(self, X, y):
        offset = np.log(X[self.anchor].to_numpy(dtype=float))
        self.inner_ = clone(self.inner).fit(X, np.log(np.asarray(y, dtype=float)) - offset)
        return self

    def predict(self, X):
        return np.exp(self.inner_.predict(X) + np.log(X[self.anchor].to_numpy(dtype=float)))


def _hgb_inner(loss: str, max_iter: int = 400, max_leaf_nodes: int = 31, learning_rate: float = 0.05):
    pre = ColumnTransformer([("cat", _onehot(), CATEGORICAL_V2), ("num", "passthrough", NUMERIC_V2 + BOOLEAN)])
    hgb = HistGradientBoostingRegressor(loss=loss, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                                        learning_rate=learning_rate, min_samples_leaf=20, l2_regularization=1.0,
                                        early_stopping=False, random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("hgb", hgb)])


def _ridge_inner(alpha: float = 3.0):
    pre = ColumnTransformer([
        ("cat", _onehot(), CATEGORICAL_V2),
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()),
         NUMERIC_V2 + BOOLEAN),
    ])
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=alpha))])


CANDIDATES_COMPS = {
    "comps_ridge": lambda: CompsResidualModel(_ridge_inner()),
    "comps_hgb_l2": lambda: CompsResidualModel(_hgb_inner("squared_error")),
    "comps_hgb_l1": lambda: CompsResidualModel(_hgb_inner("absolute_error")),
    "comps_hgb_l1_deep": lambda: CompsResidualModel(_hgb_inner("absolute_error", 800, 63, 0.03)),
}

FEATURE_SETS = {
    "v1": {"features": FEATURES, "candidates": CANDIDATES, "comps": False},
    "comps": {"features": FEATURES_V2 + [ANCHOR], "candidates": CANDIDATES_COMPS, "comps": True},
}


# ----------------------------------------------------------------------------- metrics
def error_metrics(y_true, y_pred, prefix: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ae = np.abs(y_pred - y_true)
    if len(y_true) == 0:
        return {f"{prefix}_mae": None, f"{prefix}_median_ae": None, f"{prefix}_mape": None,
                f"{prefix}_within_25pct": None, f"{prefix}_bias": None}
    return {
        f"{prefix}_mae": float(ae.mean()),
        f"{prefix}_median_ae": float(np.median(ae)),
        f"{prefix}_mape": float(np.mean(ae / y_true)),
        f"{prefix}_within_25pct": float(np.mean(ae / y_true <= 0.25)),
        f"{prefix}_bias": float(np.mean(y_pred - y_true)),  # > 0: over-pricing on average
    }


def metrics_rows(df: pd.DataFrame, model_version: str, model_name: str, split: str, scope: str,
                 group_cols=("city",)) -> list[dict]:
    """Model vs baseline metrics per city and for all cities together."""
    rows = []
    groups = [(g, d) for g, d in df.groupby(list(group_cols))] + [(("ALL",), df)]
    for key, d in groups:
        key = key if isinstance(key, tuple) else (key,)
        row = {"model_version": model_version, "model_name": model_name, "split": split, "scope": scope,
               "city": key[0], "n": int(len(d))}
        row.update(error_metrics(d["price"], d["predicted_price"], "model"))
        row.update(error_metrics(d["price"], d["baseline_price"], "baseline"))
        if row["baseline_mae"]:
            row["mae_improvement_pct"] = float(1 - row["model_mae"] / row["baseline_mae"])
        else:
            row["mae_improvement_pct"] = None
        rows.append(row)
    return rows


# ----------------------------------------------------------------------------- artifacts
def model_dir(artifacts_root: str, version: str) -> str:
    return os.path.join(artifacts_root, "models", version)


def save_artifact(artifacts_root: str, version: str, payload: dict, overwrite: bool = False) -> str:
    d = model_dir(artifacts_root, version)
    path = os.path.join(d, "model.joblib")
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"model {version} already exists at {path}; models are never overwritten "
                              f"(pass overwrite=true or use a new model_version)")
    os.makedirs(d, exist_ok=True)
    payload = {**payload, "sklearn_version": sklearn.__version__,
               "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    joblib.dump(payload, path)
    with open(os.path.join(d, "card.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in payload.items() if k not in ("model", "baseline")}, f, indent=2, default=str)
    return path


def load_artifact(artifacts_root: str, version: str) -> dict:
    return joblib.load(os.path.join(model_dir(artifacts_root, version), "model.joblib"))


def prediction_frame(pdf: pd.DataFrame, pred: np.ndarray, base: np.ndarray, model_version: str,
                     city_role: dict) -> pd.DataFrame:
    out = pdf[["listing_id", "city", "split", "is_trainable", "price_class", "room_type", "neighbourhood",
               "accommodates", "bedrooms", "monthly_stay", "latitude", "longitude", "price"]].copy()
    out = out.rename(columns={"price": "actual_price"})
    out["monthly_stay"] = out["monthly_stay"].fillna(False).astype(bool)
    out.insert(0, "model_version", model_version)
    out["city_role"] = out["city"].map(city_role).fillna("new")
    out["predicted_price"] = np.round(pred, 2)
    out["baseline_price"] = np.round(base, 2)
    out["error"] = out["predicted_price"] - out["actual_price"]  # > 0: model over-prices
    out["abs_error"] = out["error"].abs()
    out["pct_error"] = out["error"] / out["actual_price"]
    out["predicted_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return out


# ----------------------------------------------------------------------------- drift
def drift_report(train: pd.DataFrame, new: pd.DataFrame, features: list[str]) -> dict:
    """How much of a new city's input the model has never seen.

    Categorical: share of rows whose level does not occur in training.
    Numeric: share of rows outside the training [min, max].
    """
    report = {}
    for c in features:
        if c not in new.columns or c not in train.columns:
            continue
        if c in CATEGORICAL:
            seen = set(train[c].dropna().unique())
            report[c] = {"unseen_share": round(float((~new[c].isin(seen)).mean()), 4)}
        else:
            lo, hi = train[c].min(), train[c].max()
            v = new[c].dropna()
            report[c] = {"out_of_range_share": round(float(((v < lo) | (v > hi)).mean()) if len(v) else 0.0, 4)}
    return report
