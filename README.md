# iHARP Coastal Flooding Prediction – Reduced Feature Submission

This repository contains training code, experiments, and the final Codabench submission for the **iHARP ML Challenge 2 – Coastal Flooding Prediction**.

The objective of the challenge is to predict whether a flooding event will occur in the **14 days following a 7-day historical sea-level window** for US East Coast coastal stations.

---

## Repository Structure

For Codabench submission, only the following are required:

```
submissions/
  model.py
  final_reduced_model.pkl   (optional if using pretrained weights)
  requirement.txt
```

> IMPORTANT  
> Do NOT zip the entire repository when submitting to Codabench.  
> Only include the required submission files.

---

### Structure of This Repository

```
coastal-flooding-events-prediction/
│
├── NEUSTG_19502020_12stations.mat
├── Seed_Coastal_Stations.txt
├── Seed_Coastal_Stations_Thresholds.mat
├── Seed_Historical_Time_Intervals.txt
│
├── final_reduced_model.ipynb
│
├── submissions/
│   ├── model.py
│   ├── final_reduced_model.pkl
│   └── reduced_model_submission.zip
│
├── requirement.txt
├── LICENSE
└── README.md
```

---

## Problem Overview

For each station:

- Input: 7-day historical sea-level window  
- Output: Binary prediction for the next 14 days  
- Target = 1 if **any flooding day occurs** within the 14-day forecast horizon  

Flooding is defined as:

```
daily_max_sea_level > station_minor_flood_threshold
```

The model is trained on 9 coastal stations and evaluated on 3 unseen stations to test spatial generalization.

---

## Data Processing Pipeline

### 1. Hourly → Daily Aggregation

Hourly sea-level measurements are aggregated into:

- Daily mean
- Daily standard deviation
- Daily maximum

### 2. Flood Label Construction

```
flood = 1 if daily_max > threshold
flood = 0 otherwise
```

Official station thresholds are used when available.  
Fallback thresholds (mean + 1.5 × std) are used for unseen stations.

### 3. Window Construction

- 7 historical days  
- 14 future prediction days  
- Window-level binary label  

To ensure ingestion runtime safety:

- Windows are subsampled using a fixed stride  
- Total training windows are capped  

---

## Feature Engineering (Reduced Feature Model)

Feature selection was performed via model-based importance pruning.

### Flood Memory Features

- flood_lag_1  
- flood_lag_2  
- flood_lag_3  
- flood_lag_7  
- flood_lag_14  
- flood_count_7  
- flood_count_14  
- flood_count_30  

### Threshold-Relative Magnitude Features

- max_lag_1  
- max_lag_2  
- max_lag_3  
- max_lag_7  
- max_lag_14  
- max_minus_threshold  
- distance_ratio  

Low-importance seasonality and long rolling statistics were removed to improve generalization and runtime performance.

---

## Model

A single global classifier is trained:

```
XGBClassifier
```

Key hyperparameters:

- n_estimators = 260  
- max_depth = 4  
- learning_rate = 0.03  
- subsample = 0.8  
- colsample_bytree = 0.8  
- min_child_weight = 3  
- gamma = 0.1  
- reg_lambda = 2.0  
- scale_pos_weight = class imbalance adjusted  

The model predicts a **window-level binary probability**.

---

## Threshold Calibration

Since Codabench thresholds probabilities at 0.5:

1. Training data is split into train/validation.
2. The optimal F1 threshold is selected.
3. Probabilities are shifted so that 0.5 corresponds to the tuned threshold.

This improves F1 while remaining compliant with the official scoring pipeline.

---

## Installation & Running

### Install Dependencies

Using pip:

```
pip install -r requirement.txt
```

Or using conda:

```
conda create -n coastal-flood python=3.11
conda activate coastal-flood
pip install -r requirement.txt
```

---

## Running the Notebook

```
jupyter lab
```

Open:

```
final_reduced_model.ipynb
```

This notebook contains:

- Data loading  
- Feature engineering  
- Feature importance analysis  
- Reduced feature retraining  
- Validation evaluation  

---

## Codabench Submission

The ingestion-compatible entry point is:

```
submissions/model.py
```

Executed as:

```
python -m model \
  --train_hourly <csv> \
  --test_hourly <csv> \
  --test_index <csv> \
  --predictions_out predictions.csv
```

The script:

- Trains the model  
- Calibrates probabilities  
- Writes `predictions.csv` to the required output path  

---

## Reproducibility

Experiments were conducted using:

- Python 3.11  
- pandas  
- numpy  
- xgboost  
- scikit-learn  
- scipy  

See `requirement.txt` for full dependency list.

---

## License

See `LICENSE` file.
