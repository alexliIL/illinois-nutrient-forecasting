# Illinois Nutrient Forecasting

Forecasting nutrient (phosphorus, nitrate) concentration spikes in Illinois
rivers, 1–7 days ahead, using USGS gage records and Daymet weather data.

## Problem

Given information available at day *t* on a river, predict whether a
nutrient spike will occur on any day in *t+1 .. t+7*, evaluated on rivers
the model was **not** trained on. Spikes (fertilizer-driven runoff events)
matter because they feed downstream algae blooms and are when water
utilities need to react — a spike detected only after the fact is not
actionable.

This is set up as an **ablation study**: start with the minimum viable
feature set and add feature groups one at a time, measuring the PR-AUC
improvement at each stage.

- **M1** — river-state only (lagged concentration/flow, rolling means, seasonality)
- **M2** — M1 + Daymet weather (antecedent rainfall, dry-spell length, temperature)
- **M3** — + watershed/geography features (not yet implemented)
- **M4** — + satellite imagery (not yet implemented)

## Data (`data/`)

| file | description |
|---|---|
| `Illinois_nutrient_and_sediment_concentrations_wy2016-2021.csv` | Daily USGS records for 8 gages, WY2016–2021: flow (`q`), `no3_mg_per_l`, `total_p_mg_per_l`, `ssc_mg_per_l`, plus geometric standard error columns (concentrations are log-normal). |
| `site_coords.csv` | Lat/long for each of the 8 gages (via USGS NWIS). |
| `weather_daymet.csv` | Daily precip/tmax/tmin per site, 1980–2021+, pulled from Daymet at each gage's coordinates. |

## Source (`src/`)

| file | description |
|---|---|
| `get_weather.py` | One-off fetch script: pulls site coordinates from NWIS and daily Daymet weather per site, writes the two data files above. |
| `nutrient_spike_m1.py` | M1 baseline: river-state-only features, leave-one-river-out evaluation, phosphorus target. |
| `nutrient_spike_m2.py` | M2: adds weather features to M1, prints the M1 vs M2 PR-AUC comparison for phosphorus. |
| `nutrient_spike_m2_both.py` | Same M1 vs M2 ablation, run for **both** phosphorus and nitrate, for a side-by-side comparison. |

## Methodology notes

- **Target definition**: a "spike" is a day where log-concentration exceeds
  a per-river 90th-percentile threshold. The threshold is computed with an
  **expanding (causal) window** — using only data up to and including that
  day — so it reflects what a live system could actually know, rather than
  a hindsight threshold computed over the full record.
- **Evaluation**: leave-one-river-out (train on 7 rivers, test on the 8th,
  rotate). This is standard practice in hydrology for testing
  generalization to ungauged/unseen basins ("regionalization").
- **Metric**: PR-AUC (average precision), not accuracy — spikes are a
  minority class (~25–30% of days under the 7-day forward window), and
  PR-AUC is the standard choice for imbalanced/rare-event classification.
- **Model**: `RandomForestClassifier` (`class_weight="balanced"`) — a fast,
  low-maintenance baseline appropriate while the feature set is still being
  ablated; gradient boosting or an LSTM are natural next steps once M3/M4
  are in place.

## Running

```bash
pip install pandas numpy scikit-learn dataretrieval pydaymet   # + get_weather.py deps

python src/get_weather.py           # optional: refetch data/site_coords.csv, data/weather_daymet.csv
python src/nutrient_spike_m1.py     # M1 baseline
python src/nutrient_spike_m2.py     # M1 vs M2, phosphorus
python src/nutrient_spike_m2_both.py  # M1 vs M2, phosphorus + nitrate
```

## Results (current)

Leave-one-river-out, mean PR-AUC:

| nutrient | model | PR-AUC | base rate | lift over base rate |
|---|---|---|---|---|
| phosphorus | M1 (river-state) | 0.496 | 0.284 | 1.75x |
| phosphorus | M2 (river + weather) | 0.540 | 0.284 | 1.90x |
| nitrate | M1 (river-state) | 0.486 | 0.169 | 2.88x |
| nitrate | M2 (river + weather) | 0.478 | 0.169 | 2.84x |

For phosphorus, weather adds **+0.044 PR-AUC**, consistent across all 8
rivers (paired significance confirmed via Wilcoxon signed-rank, p < 0.01).

For nitrate, weather has a small **negative** effect (**−0.008 PR-AUC**),
suggesting nitrate spikes may be driven more by baseflow/leaching dynamics
than the storm-runoff mechanism that drives phosphorus.
