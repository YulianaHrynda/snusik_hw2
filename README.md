# snusik_hw2 — Airbnb price lakehouse (US Northeast)

Group Assignment 2, team **Snusik**.

| Item | Value |
|---|---|
| Region | **US Northeast** (currency: USD) |
| City A (training) | **Boston, MA** — snapshot `2026-09-19` |
| City B (training) | **Jersey City, NJ** — snapshot `2026-09-21` |
| City C (new client, hold-out) | **New York City, NY** — snapshot `2026-09-14` |

Data: [Inside Airbnb](https://insideairbnb.com/get-the-data/) (CC BY 4.0), the detailed
`listings.csv.gz` of the latest snapshot of every city.

**Why New York as city C:** it is the region's largest and most expensive market and, unlike
Boston and Jersey City, is governed by Local Law 18 (strict short-term-rental registration), so
it is the most realistic stress test of whether a model built on two mid-size markets transfers.
