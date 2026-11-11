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
