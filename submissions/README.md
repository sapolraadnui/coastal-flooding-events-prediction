# iHARP ML Challenge 2 – Coastal Flooding Prediction Model (Reduced Feature Set)

## Overview

This submission implements a **global** coastal flooding prediction model using daily-aggregated sea level observations.

It matches the Codabench starter-kit interface:

```bash
python -m model \
  --train_hourly <csv> \
  --test_hourly <csv> \
  --test_index <csv> \
  --predictions_out <csv>
```

The submission writes a CSV with:

- `id`: row identifier from `test_index.csv`
- `y_prob`: probability that **at least one flooding day** occurs in the **next 14 days** (window-level label)

---

## Data Processing

### Hourly → Daily aggregation

Hourly sea level data is aggregated into daily statistics per station:

- `sea_level_mean`
- `sea_level_std`
- `sea_level_max`

### Flood label

A day is considered flooding if:

- `sea_level_max > flood_threshold`

Threshold strategy:

- **Official minor flooding thresholds** are used for the 12 public stations (when the station name matches).
- For unseen/hidden stations, a fallback threshold is computed as `mean + 1.5 × std` (baseline-style), primarily from training data.

---

## Window Construction

For each station:

- Historical input window: **7 days**
- Prediction window: **next 14 days**

We form a **window-level** binary target:

- `y = 1` if any day in the 14-day window floods
- `y = 0` otherwise

To keep runtime within Codabench limits, training windows are subsampled with:

- `TRAIN_STRIDE` (default: every 5th start day)
- `MAX_TRAIN_SAMPLES` cap (default: 60,000)

---

## Feature Engineering (Reduced Feature Set)

Based on feature importance pruning, the model uses a compact set of high-signal predictors:

### Flood memory
- `flood_lag_1`, `flood_lag_2`, `flood_lag_3`, `flood_lag_7`, `flood_lag_14`
- `flood_count_7`, `flood_count_14`, `flood_count_30`

### Threshold-relative magnitude
- `max_lag_1`, `max_lag_2`, `max_lag_3`, `max_lag_7`, `max_lag_14`
- `max_minus_threshold`
- `distance_ratio`

This reduced set removes low-importance/duplicate signals (e.g., seasonality terms and long rolling stats) to improve generalization and speed.

---

## Model

A single global classifier is trained:

- `XGBClassifier` (XGBoost), with regularization for OOD generalization
- Class imbalance handled via `scale_pos_weight`

Key hyperparameters:

- `n_estimators=260`
- `max_depth=4`
- `learning_rate=0.03`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=3`
- `gamma=0.1`
- `reg_lambda=2.0`

---

## Probability Calibration

Codabench typically thresholds at **0.5**. To improve F1 without altering the scorer:

1. Split training windows into train/validation (stratified)
2. Find best probability threshold on validation for F1
3. Apply a **probability shift** so that `0.5` behaves like the tuned threshold

---

## Output

The submission writes to the exact `--predictions_out` path and produces:

- `predictions.csv` with columns: `id`, `y_prob`

---

## Files to Submit

- `model.py`
- `requirements.txt`
- `README.md`

(Pretrained weights are not required; the model trains during ingestion.)

