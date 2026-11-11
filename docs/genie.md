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
4. **Evaluation rows are spelled out**: training cities on `split = 'test'` only, the new city on
   all rows, `is_trainable = true`. Otherwise Genie would report in-sample errors.
5. **Unity Catalog comments** on every gold table and on the columns that carry meaning (units,
   0–1 shares vs percentages, NULL semantics) — [`genie/01_document_gold_tables.py`](../genie/01_document_gold_tables.py).
   They survive the per-city overwrites of the pipeline.
6. **Join specs with relationship types** (`price_predictions → listings` many-to-one on
   `(city, listing_id)`) so joins never fan out.
7. **Example SQL for the three required questions** plus three more (model vs baseline, private
   rooms across cities, most over-priced listings). Genie reuses them as templates.
8. **Pre-aggregated tables come with a warning**: "never sum their medians".
9. **Stable ids** (hash of the content) so re-deploying updates the same space instead of
   duplicating instructions.

## Example questions and the answers Genie must give

Answers computed from the gold tables with exactly the example SQL in `space.yml` (model `v1`):

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

**"Where does the model under-price listings the most?"** — by mean error on test listings,
neighbourhoods with ≥ 10 test listings:

| City | Neighbourhood | Test listings | Mean error | Median error |
|---|---|---:|---:|---:|
| Boston | South Boston Waterfront | 17 | −\$83.5 | −\$40.4 |
| Boston | Back Bay | 66 | −\$80.0 | −\$29.0 |
| Jersey City | Ward D | 91 | −\$68.3 | −\$2.0 |
| Boston | Fenway | 41 | −\$65.6 | −\$42.9 |
| Boston | North End | 21 | −\$55.8 | −\$12.1 |

The gap between mean and median error shows the under-pricing is concentrated in a few expensive
listings (Ward D: median error −\$2, mean −\$68).

**"How much better is the model than the baseline?"** — test MAE \$73.6 vs \$124.7 (−40.9%);
Boston \$80.7 vs \$141.5, Jersey City \$63.3 vs \$100.3.

## Status

The space definition was accepted by the Genie API up to table validation: creation is refused
until `workspace.airbnb_gold.*` exist in Unity Catalog, which requires one run of the jobs on
Databricks (see README → "Running on Databricks").
