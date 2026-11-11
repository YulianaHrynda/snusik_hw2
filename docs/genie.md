# Genie Agent

The Genie space is configuration-as-code: [`genie/space.yml`](../genie/space.yml) holds the tables,
instructions, example SQL, join definitions and sample questions;
[`genie/deploy_genie.py`](../genie/deploy_genie.py) turns it into Genie's `serialized_space`
(version 2) and creates or updates the space through the Databricks CLI.

```bash
# 0. the gold tables must exist in Unity Catalog
databricks bundle run airbnb_train_price_model
databricks bundle run airbnb_onboard_city --params city=new-york-city,model_version=v1
# 1. table/column descriptions that Genie reads: run the notebook genie/01_document_gold_tables.py once
# 2. create / update the space (stores the space id in genie/.space_id)
python genie/deploy_genie.py
python genie/deploy_genie.py --ask "What is the median price of entire homes in City A?"
```

## Data

Only **gold** tables, all in `workspace.airbnb_gold`:

| Table | Grain | Used for |
|---|---|---|
| `listings` | one listing | counts, medians, filters by room type / neighbourhood / host |
| `city_metrics` | one city | ready-made KPIs |
| `neighbourhood_metrics` | city × neighbourhood | neighbourhood rankings |
| `price_predictions` | model version × listing | predicted vs actual, under/over-pricing |
| `model_metrics` | model version × city × split × scope | MAE of model and baseline |

## Tricks used to configure Genie

1. **Business vocabulary in the instructions.** "City A/B/C" map to the city keys; "price" without an
   aggregate means the *median*; "entire home" means `room_type = 'Entire home/apt'`; "monthly"
   means `monthly_stay`. Without this, Genie averaged prices (pulled up by luxury homes) and could
   not resolve "City A".
2. **One filter for every price statistic.** `has_valid_price = true` excludes missing and
   implausible prices — a single boolean is much easier for Genie to apply consistently than the
   underlying rule.
3. **Sign conventions written down twice** — in the instructions and as a column comment:
   `error = predicted − actual`, negative = under-priced. "Under-price the most" then becomes
   `ORDER BY avg(error) ASC`, not `DESC`.
4. **Evaluation rows are spelled out**: every city on `split = 'test'` (New York's test split was
   never trained on), `is_trainable = true`. Otherwise Genie would report in-sample errors.
5. **Unity Catalog comments** on every gold table and on the columns that carry meaning (units,
   0–1 shares vs percentages, NULL semantics) — [`genie/01_document_gold_tables.py`](../genie/01_document_gold_tables.py).
   They survive the per-city overwrites of the pipeline.
6. **Join specs with relationship types** (`price_predictions → listings` many-to-one on
   `(city, listing_id)`) so joins never fan out.
7. **Example SQL for the three required questions** plus four more (model vs baseline, private
   rooms across cities, most over-priced listings, New York by borough). Genie reuses them as templates.
8. **Model versions are named in plain words** — "first run" = `v1`, current = `v2`, and `v2_abc` only
   when asked about training on New York — so Genie does not average across versions.
9. **Pre-aggregated tables come with a warning**: "never sum their medians".
10. **Stable ids** (hash of the content) so re-deploying updates the same space instead of
   duplicating instructions.

## Example questions and the answers Genie must give

Answers computed from the gold tables with exactly the example SQL in `space.yml`:

**"What is the median price of entire homes in City A?"** — **\$297.16** per night (Boston,
2,391 entire homes with a valid price).

**"Which neighbourhoods in City B have the most listings?"** — Jersey City wards:

| Neighbourhood | Listings |
|---|---:|
| Ward D (councilmember Michael Yun) | 533 |
| Ward F (councilmember Jermaine D. Robinson) | 479 |
| Ward C (councilmember Richard Boggiano) | 453 |
| Ward E (councilmember James Solomon) | 384 |
| Ward B (councilmember Mira Prinz-Arey) | 321 |
| Ward A (councilmember Denise Ridley) | 160 |

**"Where does the model under-price listings the most?"** — current model `v2`, test listings, by
neighbourhood (Boston, Jersey City) or borough (New York), areas with ≥ 10 test listings:

| City | Area | Test listings | Mean error | Median error |
|---|---|---:|---:|---:|
| New York City | Manhattan | 1,681 | −\$124.9 | −\$26.8 |
| Boston | Back Bay | 66 | −\$87.2 | −\$23.5 |
| Boston | Fenway | 41 | −\$67.1 | −\$35.1 |
| Boston | South Boston Waterfront | 17 | −\$65.1 | −\$31.5 |
| Jersey City | Ward D | 91 | −\$60.9 | −\$3.3 |

The gap between mean and median error shows the under-pricing is concentrated in a few expensive
listings (Ward D: median −\$3, mean −\$61). Asked about the *first run* (`v1`), Manhattan is at
−\$224.2 on average.

**"How much better is the model than the baseline?"** — test MAE, baseline → model:

| | `v1` | `v2` |
|---|---:|---:|
| Boston + Jersey City | \$124.7 → \$73.6 (−40.9%) | \$124.7 → \$76.1 (−38.9%) |
| New York City | \$164.2 → \$143.2 (−12.8%) | \$164.2 → \$114.7 (−30.1%) |

**"How well does the model price New York, by borough?"** — test MAE `v1` → `v2`: Manhattan
\$231.1 → \$174.9, Brooklyn \$82.0 → \$72.4, Queens \$66.0 → \$62.6, Bronx \$52.0 → \$53.5,
Staten Island \$40.9 → \$42.7.

**"Compare the median price of private rooms across cities"** — Boston \$118.70, New York
\$114.56, Jersey City \$80.32.

## Status — deployed and tested on Databricks

The space *Airbnb US Northeast — prices and price model (Snusik)* was created with
`python genie/deploy_genie.py` (id in [`genie/.space_id`](../genie/.space_id)) after the column
comments notebook ran. Asked through the API, it generated exactly the example SQL and answered:

| Question | Genie's answer (Databricks tables) |
|---|---|
| What is the median price of entire homes in City A? | \$297.16 per night, 2,391 listings |
| Which neighbourhoods in City B have the most listings? | Ward D 533, Ward F 479, Ward C 453, … |
| Where does the model under-price listings the most? | Manhattan: mean error −\$127.7 (median −\$27.0), then Back Bay −\$84.2 and South Boston Waterfront −\$65.0 (`v2`, test listings) |
| How much better is the model than the baseline in each city? | `v2`: Boston 41.5%, Jersey City 35.8%, New York 29.9% |
| How well did the first run of the model price New York, by borough? | `v1`: Manhattan MAE \$232.9 (mean error −\$226.2) … Staten Island \$41.4 |
| What is the MAE of the current model in New York compared to the baseline? | \$115.08 vs \$164.16 |

Model-related answers differ from the tables above by up to ~2% because the Databricks run uses
the serverless scikit-learn version (see README). One fix came out of testing: Genie first called
MAE "median absolute error", so the instructions now say MAE always means the *mean* absolute error.
Genie also requires every list in the serialized space to be sorted by id — `deploy_genie.py` sorts them.
