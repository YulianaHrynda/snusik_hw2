from __future__ import annotations

import json
import os
from dataclasses import dataclass

REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cities.json")
SOURCE_URL = "https://data.insideairbnb.com/{source_path}/{snapshot_date}/data/listings.csv.gz"


@dataclass(frozen=True)
class City:
    key: str
    name: str
    source_path: str
    snapshot_date: str

    @property
    def url(self) -> str:
        return SOURCE_URL.format(source_path=self.source_path, snapshot_date=self.snapshot_date)


def load_registry(path: str = REGISTRY_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_city(key: str, snapshot_date: str = "") -> City:
    cities = load_registry()["cities"]
    if key not in cities:
        raise KeyError(f"City '{key}' is not registered in {REGISTRY_PATH}; known cities: {sorted(cities)}")
    entry = cities[key]
    return City(key=key, name=entry["name"], source_path=entry["source_path"],
                snapshot_date=snapshot_date or entry["snapshot_date"])


def parse_city_list(value: str) -> list[str]:
    value = value.strip()
    if value.startswith("["):
        return [str(v) for v in json.loads(value)]
    return [v.strip() for v in value.split(",") if v.strip()]


@dataclass(frozen=True)
class Lakehouse:
    catalog: str = "workspace"
    prefix: str = "airbnb"
    raw_root_override: str = ""
    artifacts_root_override: str = ""

    def schema(self, layer: str) -> str:
        return f"{self.catalog}.{self.prefix}_{layer}"

    def table(self, layer: str, name: str) -> str:
        return f"{self.schema(layer)}.{name}"

    @property
    def raw_root(self) -> str:
        return self.raw_root_override or f"/Volumes/{self.catalog}/{self.prefix}_bronze/raw"

    @property
    def artifacts_root(self) -> str:
        return self.artifacts_root_override or f"/Volumes/{self.catalog}/{self.prefix}_gold/artifacts"


def is_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def read_params(dbutils, **defaults: str) -> dict:
    for name, default in defaults.items():
        dbutils.widgets.text(name, default)
    return {name: dbutils.widgets.get(name).strip() for name in defaults}


def lakehouse_from(params: dict) -> Lakehouse:
    return Lakehouse(
        catalog=params.get("catalog") or "workspace",
        raw_root_override=params.get("raw_root", ""),
        artifacts_root_override=params.get("artifacts_root", ""),
    )
