# Price model (cities A and B)

Code: [`src/airbnb/modelling.py`](../src/airbnb/modelling.py),
[`src/model/01_train_price_model.py`](../src/model/01_train_price_model.py) (job
`airbnb_train_price_model`). Model version **`v1`** is the end-of-week-1 model, used unchanged
for the first city C run.

## Setup

| | |
|---|---|
| Target | `price` — advertised price per night, USD. The model is fitted on **log price** and predictions are converted back. |
| Rows | listings with a usable price: `price_class ∈ {normal, extreme}`. Listings without a price (1,118) cannot be evaluated; the 7 `implausible` prices (≥ \$750 per guest, EDA §5) are excluded from training and from the main metric, and reported separately. |
| Split | fixed in gold before any model was trained: `pmod(xxhash64(listing_id), 100) < 20` → test. Boston 2,857 train / 626 test, Jersey City 1,658 / 432. |
| Baseline | median **training** price for the same `room_type` and `accommodates`, pooled over both cities (falls back to the room-type median, then the global median, for unseen combinations). |
| Metric | MAE in USD on the test split, per city and for both together. Also median absolute error, MAPE, share within ±25%, mean error (bias). |
| Model choice | 5-fold cross-validation **on the training split only**; the test split is used once, for the chosen model. |

## Features

| Group | Features | Why |
|---|---|---|
| Unit | `room_type`, `property_type`, `accommodates`, `bedrooms`, `beds`, `bathrooms`, `bath_shared` | size and privacy are what a guest pays for |
| Stay rules | `minimum_nights`, `monthly_stay` (≥ 28 nights) | monthly-priced listings advertise less than half the nightly price (EDA §4.2) |
| Location | `city`, `neighbourhood`, `latitude`, `longitude` | location premium within a city |
| Amenities | amenity count + 13 flags (air conditioning, pool, hot tub, gym, elevator, free parking, washer, dishwasher, workspace, self check-in, outdoor space, skyline/water view, pets allowed) | quality signals visible on the listing |
| Host | `host_is_superhost`, `host_identity_verified`, `host_listings_count`, `host_years` | professional operators price differently |
| Reputation | `number_of_reviews`, `has_reviews`, review scores (rating, cleanliness, location, value) | quality as perceived by guests |

Not used (EDA §7): every `price_quote_*` column and `estimated_revenue_l365d` (computed from the
price), and booking outcomes (`availability_*`, `estimated_occupancy_l365d`, recent review
counts, `reviews_per_month`) that react to the price. Categorical levels seen fewer than 10 times
in training are grouped as "infrequent"; unseen levels go to the same bucket.

## Model choice (5-fold CV MAE on the training split, USD)

| Candidate | CV MAE | ± std |
|---|---:|---:|
| Baseline (median by room type × guests) | 113.96 | 3.07 |
| Ridge on log price (one-hot + scaled numerics) | 81.92 | 1.36 |
| Gradient boosting, log price, squared loss | 71.50 | 1.71 |
| Gradient boosting, log price, absolute loss | 71.62 | 1.99 |
| **Gradient boosting, log price, absolute loss, 800 trees × 63 leaves** | **70.68** | 1.83 |

Chosen: `HistGradientBoostingRegressor(loss="absolute_error")` on log price. Absolute loss on log
price predicts the conditional **median** of log price, and the median is preserved by `exp`, so
the model predicts a median price — the point estimate that minimises MAE. The log transform makes
a \$50 miss on a \$100 room count as much as a \$500 miss on a \$1,000 home during training.

## Test-set results (`v1`)

| City | Test listings | Baseline MAE | **Model MAE** | Improvement | Median abs. error | MAPE | Within ±25% | Mean error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Boston (A) | 626 | \$141.47 | **\$80.72** | −42.9% | \$41.3 | 23.9% | 66.6% | −\$26.1 |
| Jersey City (B) | 432 | \$100.33 | **\$63.35** | −36.9% | \$26.7 | 21.3% | 66.9% | −\$25.1 |
| **A + B** | 1,058 | \$124.67 | **\$73.63** | **−40.9%** | \$34.7 | 22.8% | 66.7% | −\$25.7 |

Including the implausible prices the main metric excludes (1 extra test listing, in Jersey City):
A + B MAE \$75.17 vs baseline \$126.07.

Test MAE (\$73.6) is close to the CV estimate (\$70.7), so the CV did not over-fit the model choice.
Training MAE is lower (\$38.3): the trees memorise part of the training set, which is why only
test numbers are reported.

**By room type (test MAE, model vs baseline):** Boston entire homes \$89.9 vs \$151.1, private rooms
\$55.7 vs \$111.1, hotel rooms \$91.9 vs \$180.4; Jersey City entire homes \$85.6 vs \$131.4,
private rooms \$21.0 vs \$41.4. The model is better than the baseline in every segment.

**Bias.** The mean error is about −\$26: the model under-prices on average. That is expected for a
median predictor on right-skewed prices (mean > median), and the biggest misses are large luxury
homes priced far above similar listings.

## What drives the predictions (permutation importance on the test split)

Increase in test MAE when a feature is shuffled:

| Feature | +MAE, \$ |
|---|---:|
| `minimum_nights` | 39.9 |
| `room_type` | 17.9 |
| `accommodates` | 13.8 |
| `bedrooms` | 13.4 |
| `bathrooms` | 11.2 |
| `latitude` | 9.0 |
| `longitude` | 8.8 |
| `review_scores_location` | 6.2 |
| `bath_shared` | 3.8 |
| `has_elevator` | 2.9 |

Stay length dominates (the monthly-pricing effect), then size and privacy, then location. Host
and amenity features add little each.

## Transfer risks we see before city C

Written before city C was loaded:

* `city`, `neighbourhood`, `latitude` and `longitude` are city-specific. For a new city the city
  and neighbourhood go to the "infrequent/unknown" bucket and the coordinates are outside anything
  the trees have seen, so the model has no location knowledge there.
* The model never saw a market with a different price level; tree models cannot predict outside
  the range of training prices.
* The baseline and the outlier rule (\$750 per guest) were calibrated on A and B only.
