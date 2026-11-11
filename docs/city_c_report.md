# The new client: New York City (city C)

City C is **New York City**, Inside Airbnb snapshot `2026-09-14`, 29,918 listings. It was not
loaded or explored before the pipeline and model `v1` for Boston and Jersey City were committed
(see the git history: the NYC entry in `cities.json` arrives after the model, dashboard and Genie
commits).

## 1. First run — unchanged pipeline, unchanged model `v1`

The only change was the one configuration entry the pipeline is designed for — the city in
[`src/airbnb/cities.json`](../src/airbnb/cities.json):

```json
"new-york-city": {"name": "New York City", "source_path": "united-states/ny/new-york-city", "snapshot_date": "2026-09-14"}
```

Then the onboarding job (`airbnb_onboard_city`, run locally with
`python scripts/run_local.py onboard --city new-york-city --model-version v1`): city pipeline →
score with the saved `v1` model. Log: [`city_c/first_run.log`](city_c/first_run.log).

### Did it run? Yes — nothing failed

| Step | Result |
|---|---|
| Download | the file was downloaded into the landing zone (16 MB gzip) |
| Bronze | 29,918 rows, the same 90 columns as Boston and Jersey City |
| Silver | 0 parse failures in every checked column, 0 duplicates; price classes: 19,180 normal, 115 extreme, **111 implausible**, **10,512 missing** |
| Gold | 29,918 listings, 223 neighbourhoods, 1 city row; 15,481 / 3,814 trainable listings in train / test split |
| Scoring | 29,918 predictions written for `v1`, metrics written |

### What came out — the MAE on city C

Every NYC listing is out-of-sample for `v1`, so it is evaluated on all 19,295 listings with a valid
price (and, for comparability with A and B, on the same hash-based 20% test split).

| | Listings | Baseline MAE | `v1` MAE | `v1` vs baseline | Mean error |
|---|---:|---:|---:|---:|---:|
| Boston + Jersey City (test) | 1,058 | \$124.67 | **\$73.63** | −40.9% | −\$25.7 |
| **New York City, all valid prices** | 19,295 | \$159.00 | **\$137.58** | −13.5% | **−\$122.0** |
| New York City, test split | 3,814 | \$164.16 | \$143.16 | −12.8% | −\$127.6 |

The model still beats the baseline in New York, but its error is **1.9× its error at home**, the
advantage over the baseline shrinks from 41% to 13%, and it under-prices by \$122 a night on average.

### Why — what the first run tells us

**1. The pipeline ran but did not warn.** A third of NYC's target is missing (35% vs 10–20% in A
and B), 81% of listings are monthly rentals, 111 prices are implausible (vs 7) and a column that
was empty in A and B is full in C. All of that went through silently; we found it by looking.

| | Boston | Jersey City | New York |
|---|---:|---:|---:|
| Listings without price | 20.2% | 10.1% | **35.1%** |
| Monthly stays (min. 28+ nights) | 41.6% | 42.2% | **80.7%** |
| Priced listings that are short-stay | 67.8% | 62.2% | **26.3%** |
| Median price, short stay | \$311 | \$298 | **\$373** |
| Median price, monthly stay | \$134 | \$107 | \$146 |
| Implausible prices | 3 | 4 | 111 |
| `neighbourhood_group_cleansed` filled | 0% | 0% | **100%** (borough) |

New York is a different *market*: Local Law 18 effectively bans stays under 30 nights without a
registration, so 23,222 listings have exactly a 30-night minimum. The few short-stay listings that
remain are expensive hotel-like units.

**2. The error is concentrated where the price level is new.** By borough:

| Borough | Listings | Actual median | Predicted median | `v1` MAE | Mean error |
|---|---:|---:|---:|---:|---:|
| Bronx | 764 | \$108 | \$112 | \$47 | −\$16 |
| Staten Island | 286 | \$106 | \$101 | \$48 | −\$32 |
| Queens | 3,234 | \$126 | \$123 | \$59 | −\$32 |
| Brooklyn | 6,352 | \$156 | \$135 | \$84 | −\$65 |
| **Manhattan** | 8,659 | **\$262** | \$165 | **\$217** | **−\$210** |

In the outer boroughs the model is as good as at home. In Manhattan it predicts Jersey City
prices. By segment, short-stay listings are 26% of the listings but **49% of the absolute
error** (actual median \$373, predicted \$231); hotel rooms are under-priced by \$484 on average
(actual median \$629, predicted \$280). By price band the model is fine below \$150 and compresses
everything above: listings over \$700 are predicted at a median of \$288.

**3. Location features did not transfer — and one means something else.** `city` is an unseen
level (Boston/Jersey City were the only values), all 223 neighbourhood names are unseen except
**"Chinatown"**, which the model learned in *Boston* (49 listings, median \$316): NYC's Chinatown
listings get Boston's Chinatown effect (actual mean \$405, predicted \$220). Latitude/longitude of
NYC are outside Boston's range and overlap Jersey City's, so the trees place all of New York "in
Jersey City" — which is roughly right for Queens and the Bronx and badly wrong for Manhattan.

**4. A single price-level factor explains a large part of the gap.** Multiplying every `v1`
prediction by the median actual/predicted ratio in NYC (1.29) would already cut the MAE from
\$137.6 to \$122.0 — the model's *relative* judgement (bigger, better, more private → more
expensive) transfers; its *level* does not.

## 2. What we changed, and why

Every change, in the order made. "Pipeline fix" = the pipeline should have handled it from the
start; "model change" = the model did not transfer.

| # | Kind | Change | Reason (what NYC showed) | Commit |
|---|---|---|---|---|
| 0 | configuration | add `new-york-city` to `cities.json` | the designed way to add a city — no code copied | first run |
| 1 | **pipeline fix** | carry `neighbourhood_group_cleansed` (borough) through silver → gold (`neighbourhood_group`) | schema difference: the column is 100% empty in A and B, 100% full in C; silver dropped it silently, so NYC (223 neighbourhoods) had no usable area level for the dashboard and Genie | pipeline fixes |
| 2 | **pipeline fix** | required vs optional source columns: missing required → fail with a clear message; missing optional → NULL + warning | #1 proved city files differ; a city without an optional column would have crashed silver with an analysis error | pipeline fixes |
| 3 | **pipeline fix** | silver profiles the target and the market and warns outside the A/B ranges | the first run was "green" although 35% of the target was missing; now NYC logs 4 warnings: missing price 35.1%, implausible 0.4%, monthly stays 80.7%, stale records 29.8% | pipeline fixes |
| 4 | **pipeline fix** | scoring logs an input-drift report (unseen category levels, values outside the training range) | nothing told us the model was scoring inputs it had never seen; for `v1` on NYC: `city` 100% unseen, `neighbourhood` 99.2% unseen, `latitude` 23% out of range | pipeline fixes |
| 5 | **model change** | drop `city`, `neighbourhood`, `latitude`, `longitude` | features that mean nothing — or something else — in C: unseen levels, "Chinatown" = Boston's Chinatown, coordinates placed NYC "in Jersey City" | model v2 |
| 6 | **model change** | market comps from the city's own training-split listings (20 nearest same-segment, 50 nearest overall; never the listing itself or its host's listings) and a target relative to them, `log(price / comp price)` | a price level the model never saw: Manhattan and short-stay prices are above anything learnt in A/B, and trees cannot extrapolate | model v2 |
| 7 | model change (support) | `host_id` added to `gold.listing_features` | needed by #6 to exclude a host's own listings from its comps | model v2 |

Checked and deliberately **not** changed:

* **The implausible-price rule (≥ \$750 per guest).** It flags 111 NYC listings (0.4%, vs 0.1% in
  A/B). We read them: \$12k–31k placeholder prices and \$800–1,300-a-night *single* private rooms
  with no reviews and a 30-night minimum — errors, not luxury. The rule transfers; it now warns.
* **The monthly threshold (28 nights).** NYC's 30-night minimum is correctly classified as monthly.
* **The baseline.** By definition it stays the A+B training medians.

The drift report also exposed a limit of simple range checks: NYC longitudes lie *between* Boston's
and Jersey City's, so they are "in range" for a two-city training set although no training listing
is anywhere near them.

## 3. The numbers side by side

MAE in USD per night. Training cities: their 20% test split. New York: the same hash-based 20% test
split (3,814 listings) — it was never used for training by `v1` or `v2`.

| | Boston + Jersey City (test) | New York City (test) |
|---|---:|---:|
| **Baseline** (median of A+B training prices by room type × guests) | \$124.67 | **\$164.16** |
| **Model `v1`** — end of week 1, first run on C | **\$73.63** | **\$143.16** |
| **Model `v2`** — after the changes (still trained on A+B only) | \$76.13 | **\$114.70** |
| *reference:* baseline rule fitted on NYC's own training split | – | \$153.42 |
| *reference:* `v2` recipe retrained with NYC's training split (`v2_abc`) | \$78.94 | \$87.58 |

On *all* 19,295 NYC listings with a valid price: baseline \$159.00, `v1` \$137.58, `v2` \$110.73.

Other views of the same predictions (test splits):

| | MAE | Median abs. error | MAPE | Within ±25% | Mean error |
|---|---:|---:|---:|---:|---:|
| `v1` on A+B | \$73.63 | \$34.7 | 22.8% | 66.7% | −\$25.7 |
| `v1` on NYC | \$143.16 | \$49.8 | 35.2% | 40.9% | −\$127.6 |
| `v2` on NYC | \$114.70 | \$41.7 | 32.4% | 48.4% | −\$74.1 |
| `v2_abc` on NYC (reference) | \$87.58 | \$32.5 | 25.4% | 60.8% | −\$40.0 |

## 4. Explaining the gap

The first-run gap is \$143.16 − \$73.63 = **\$69.5 per night**. \$2.5 of it is what `v2` gives up at
home (A+B: \$73.63 → \$76.13); the remaining \$67.0 breaks down into three parts:

1. **\$28.5 — features and price level that did not transfer** (`v1` → `v2`). Once the model stops
   relying on Boston/Jersey City location features and anchors on what comparable NYC listings
   charge, the systematic under-pricing halves (−\$128 → −\$74) and Manhattan's MAE drops from \$231 to
   \$175. That is the part a better *transfer* design fixes without any NYC training.
2. **\$27.1 — what can only be learnt from NYC itself** (`v2` → `v2_abc`). The same recipe trained
   with NYC listings reaches \$87.58. The missing knowledge is specific to New York's market structure:
   under Local Law 18 stays under 30 nights need a host registration, so the 19% of listings that
   still take short stays are mostly registered units and hotel rooms, priced at a premium over the
   monthly rentals around them, and Boston and Jersey City
   have almost no examples (84 hotel rooms, no comparable regulation). Even `v2` predicts NYC
   short stays at a median of \$301 against an actual \$380, and hotel rooms \$420 against \$577.
3. **\$11.5 — NYC is a harder market** (`v2_abc` on NYC vs `v2` on A+B). Its prices are more spread
   out: with its own training data the model is nearly as good in *relative* terms (MAPE 25.4% vs
   22.8%), but the same relative error on a \$400–\$7,800 Manhattan listing is many more dollars. Even
   `v2_abc` has an MAE of \$195 on Manhattan short stays, against \$34–55 on monthly rentals in the
   outer boroughs.

**What we would tell the client.** On day one we can price New York's outer boroughs and monthly
rentals about as well as Boston and Jersey City (Queens, Bronx, Staten Island: \$43–63 MAE with `v2`).
Manhattan short-stay and hotel prices are not reliable without NYC training data: retraining on the
NYC snapshot (the `v2_abc` experiment) brings the city to \$87.58, and that is what we would ship
after the first week with the client.

## How to reproduce

```bash
python scripts/run_local.py train --train-cities boston,jersey-city --model-version v1 --feature-set v1
python scripts/run_local.py onboard --city new-york-city --model-version v1   # first run
python scripts/run_local.py train --train-cities boston,jersey-city --model-version v2 --feature-set comps
python scripts/run_local.py onboard --city new-york-city --model-version v2
python scripts/run_local.py train --skip-etl --train-cities boston,jersey-city,new-york-city \
    --model-version v2_abc --feature-set comps --model-name comps_hgb_l1                 # reference
```

On Databricks the same steps are the jobs `airbnb_train_price_model` (parameters `train_cities`,
`model_version`, `feature_set`) and `airbnb_onboard_city` (`city`, `model_version`).
