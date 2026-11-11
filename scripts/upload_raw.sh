#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

city="${1:?usage: scripts/upload_raw.sh <city-key> [catalog]}"
catalog="${2:-workspace}"

read -r source_path snapshot_date < <(python3 -c "
import json, sys
c = json.load(open('src/airbnb/cities.json'))['cities'].get('$city')
if c is None:
    sys.exit('city $city is not in src/airbnb/cities.json')
print(c['source_path'], c['snapshot_date'])")

local="data/raw/$city/$snapshot_date/listings.csv.gz"
if [ ! -s "$local" ]; then
  mkdir -p "$(dirname "$local")"
  curl -fsSL -o "$local" "https://data.insideairbnb.com/$source_path/$snapshot_date/data/listings.csv.gz"
fi

databricks schemas create airbnb_bronze "$catalog" >/dev/null 2>&1 || true
databricks volumes create "$catalog" airbnb_bronze raw MANAGED >/dev/null 2>&1 || true

remote="dbfs:/Volumes/$catalog/airbnb_bronze/raw/$city/$snapshot_date"
databricks fs mkdir "$remote"
databricks fs cp --overwrite "$local" "$remote/listings.csv.gz"
echo "uploaded $local -> $remote/listings.csv.gz"
