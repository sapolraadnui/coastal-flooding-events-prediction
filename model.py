#!/usr/bin/env python3
"""
iHARP ML Challenge 2 - Coastal Flooding Prediction (Codabench submission)

This model.py matches the STARTER KIT ingestion interface:

  python -m model --train_hourly <csv> --test_hourly <csv> --test_index <csv> --predictions_out <csv>

Pipeline:
- Read hourly train/test CSVs
- Aggregate to daily (mean/std/max)
- Create flood labels using station thresholds (official for known stations; fallback for unseen stations)
- Engineer daily features
- Use a REDUCED feature set (keep_features) based on feature-importance pruning
- Train a single global classifier on subsampled training windows (runtime safe)
- Calibrate decision threshold on a validation split and *shift* probabilities so that the scorer's 0.5
  threshold behaves like the tuned threshold (boosts F1 without changing scoring code)
- Produce predictions aligned to test_index.csv and write to the exact --predictions_out path

Output format:
- CSV with columns: id, y_prob
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    _HAS_XGB = False

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


# -------------------------
# Configuration
# -------------------------
HIST_DAYS = 7
FUTURE_DAYS = 14

# Runtime safety knobs
TRAIN_STRIDE = 5          # keep 1 window every N days per station
MAX_TRAIN_SAMPLES = 60000 # hard cap across all stations

# Reduced feature set from importance pruning (will be intersected with available columns)
KEEP_FEATURES = [
    # flood memory
    "flood_lag_1", "flood_lag_2", "flood_lag_3", "flood_lag_7", "flood_lag_14",
    "flood_count_7", "flood_count_14", "flood_count_30",
    # magnitude / threshold-relative
    "distance_ratio", "max_minus_threshold",
    "max_lag_1", "max_lag_2", "max_lag_3", "max_lag_7", "max_lag_14",
]

# Official thresholds for the 12 public stations (minor flooding)
OFFICIAL_THRESHOLDS = {
    "Annapolis": 2.104,
    "Atlantic_City": 3.344,
    "Charleston": 2.98,
    "Eastport": 8.071,
    "Fernandina_Beach": 3.148,
    "Lewes": 2.675,
    "Portland": 6.267,
    "Sandy_Hook": 2.809,
    "Sewells_Point": 2.706,
    "The_Battery": 3.192,
    "Washington": 2.673,
    "Wilmington": 2.423,
}


# -------------------------
# Helpers
# -------------------------
def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    return df


def daily_aggregate(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """Hourly -> daily aggregation with stable columns."""
    df = _ensure_datetime(df_hourly)
    df["date"] = df["time"].dt.floor("D")

    agg_dict = {
        "sea_level_mean": ("sea_level", "mean"),
        "sea_level_std": ("sea_level", "std"),
        "sea_level_max": ("sea_level", "max"),
    }
    if "latitude" in df.columns:
        agg_dict["latitude"] = ("latitude", "first")
    if "longitude" in df.columns:
        agg_dict["longitude"] = ("longitude", "first")

    daily = (
        df.groupby(["station_name", "date"])  # type: ignore[arg-type]
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["station_name", "date"])
        .reset_index(drop=True)
    )
    return daily


def build_thresholds_from_train(train_hourly: pd.DataFrame) -> pd.DataFrame:
    """Use official thresholds when available; fallback to mean + 1.5*std from training."""
    stats = (
        train_hourly.groupby("station_name")["sea_level"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats["fallback_threshold"] = stats["mean"] + 1.5 * stats["std"]

    def pick_thr(row):
        st = row["station_name"]
        if st in OFFICIAL_THRESHOLDS:
            return float(OFFICIAL_THRESHOLDS[st])
        return float(row["fallback_threshold"])

    stats["flood_threshold"] = stats.apply(pick_thr, axis=1)
    return stats[["station_name", "flood_threshold"]]


def add_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute all candidate daily features; final model will use KEEP_FEATURES subset."""
    d = daily.sort_values(["station_name", "date"]).reset_index(drop=True).copy()

    # lags (max, mean, flood)
    for lag in [1, 2, 3, 7, 14]:
        d[f"max_lag_{lag}"] = d.groupby("station_name")["sea_level_max"].shift(lag)
        d[f"mean_lag_{lag}"] = d.groupby("station_name")["sea_level_mean"].shift(lag)
        d[f"flood_lag_{lag}"] = d.groupby("station_name")["flood"].shift(lag)

    # recent flood counts
    for w in [7, 14, 30]:
        d[f"flood_count_{w}"] = d.groupby("station_name")["flood"].shift(1).rolling(w).sum()

    # threshold-relative
    d["max_minus_threshold"] = d["sea_level_max"] - d["flood_threshold"]
    d["distance_ratio"] = d["max_minus_threshold"] / (d["flood_threshold"] + 1e-6)

    return d


def build_windows_for_training(daily_feat: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Train samples: features at end of 7-day history; label=any flood in next 14 days."""
    X_list, y_list = [], []
    n = 0

    for _, grp in daily_feat.groupby("station_name"):
        grp = grp.sort_values("date").reset_index(drop=True)
        max_i = len(grp) - HIST_DAYS - FUTURE_DAYS + 1
        if max_i <= 0:
            continue

        for i in range(0, max_i, TRAIN_STRIDE):
            idx_feat = i + HIST_DAYS - 1
            row = grp.loc[idx_feat, feature_cols]
            if row.isna().any():
                continue

            fut = grp.loc[i + HIST_DAYS : i + HIST_DAYS + FUTURE_DAYS - 1, "flood"]
            if fut.isna().any():
                continue

            X_list.append(row.to_numpy(dtype=float))
            y_list.append(int(fut.max() > 0))
            n += 1
            if n >= MAX_TRAIN_SAMPLES:
                break
        if n >= MAX_TRAIN_SAMPLES:
            break

    if not X_list:
        raise RuntimeError("No training windows were built. Check preprocessing / thresholds / NaNs.")

    return np.vstack(X_list), np.array(y_list, dtype=int)


def build_windows_for_test(daily_feat: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Build all possible test windows to match keys in test_index."""
    metas = []
    X_list = []

    for st, grp in daily_feat.groupby("station_name"):
        grp = grp.sort_values("date").reset_index(drop=True)
        max_i = len(grp) - HIST_DAYS - FUTURE_DAYS + 1
        if max_i <= 0:
            continue

        for i in range(0, max_i):
            hist_start = grp.loc[i, "date"]
            future_start = grp.loc[i + HIST_DAYS, "date"]
            idx_feat = i + HIST_DAYS - 1
            row = grp.loc[idx_feat, feature_cols]
            if row.isna().any():
                continue

            X_list.append(row.to_numpy(dtype=float))
            metas.append(
                {
                    "station_name": st,
                    "hist_start": pd.to_datetime(hist_start),
                    "future_start": pd.to_datetime(future_start),
                }
            )

    if not X_list:
        raise RuntimeError("No test windows could be built from test_hourly. Check dates / missingness.")

    meta_df = pd.DataFrame(metas)
    X = np.vstack(X_list)
    meta_df["row_ix"] = np.arange(len(meta_df), dtype=int)
    return meta_df, X


def make_key(station: pd.Series, hist_start: pd.Series, future_start: pd.Series) -> pd.Series:
    return station.astype(str) + "|" + hist_start.astype(str) + "|" + future_start.astype(str)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    train = pd.read_csv(args.train_hourly)
    test = pd.read_csv(args.test_hourly)
    index = pd.read_csv(args.test_index)

    # daily aggregation
    daily_tr = daily_aggregate(train)
    daily_te = daily_aggregate(test)

    # thresholds
    thr = build_thresholds_from_train(train)
    daily_tr = daily_tr.merge(thr, on="station_name", how="left")
    daily_te = daily_te.merge(thr, on="station_name", how="left")

    # if unseen station thresholds missing in test, fallback on test station data
    missing_thr_stations = daily_te.loc[daily_te["flood_threshold"].isna(), "station_name"].unique().tolist()
    if missing_thr_stations:
        test_stats = test.groupby("station_name")["sea_level"].agg(["mean", "std"]).reset_index()
        test_stats["flood_threshold"] = test_stats["mean"] + 1.5 * test_stats["std"]
        daily_te = daily_te.drop(columns=["flood_threshold"]).merge(
            test_stats[["station_name", "flood_threshold"]], on="station_name", how="left"
        )

    # labels (needed for lag/rolling features + training target)
    daily_tr["flood"] = (daily_tr["sea_level_max"] > daily_tr["flood_threshold"]).astype(int)
    daily_te["flood"] = (daily_te["sea_level_max"] > daily_te["flood_threshold"]).astype(int)

    # features
    feat_tr = add_daily_features(daily_tr)
    feat_te = add_daily_features(daily_te)

    # reduced feature columns
    feature_cols = [f for f in KEEP_FEATURES if f in feat_tr.columns]
    if not feature_cols:
        raise RuntimeError("KEEP_FEATURES did not match any computed feature columns.")

    # build train windows
    X_tr, y_tr = build_windows_for_training(feat_tr, feature_cols)

    # small validation split for threshold tuning
    if len(np.unique(y_tr)) > 1 and X_tr.shape[0] >= 2000:
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=42, stratify=y_tr
        )
    else:
        X_train, y_train = X_tr, y_tr
        X_val, y_val = None, None

    # train classifier
    if _HAS_XGB:
        pos = int((y_train == 1).sum())
        neg = int((y_train == 0).sum())
        spw = float(neg / max(pos, 1))

        clf = XGBClassifier(
            n_estimators=260,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=42,
            scale_pos_weight=spw,
        )
    else:
        clf = XGBClassifier(random_state=42)

    clf.fit(X_train, y_train)

    # tune threshold on validation (optimize F1) and shift probabilities
    prob_shift = 0.0
    if X_val is not None and y_val is not None and hasattr(clf, "predict_proba"):
        val_probs = clf.predict_proba(X_val)[:, 1]
        best_thr, best_f1 = 0.5, -1.0
        for thr_ in np.linspace(0.05, 0.95, 91):
            f1 = f1_score(y_val, (val_probs > thr_).astype(int))
            if f1 > best_f1:
                best_f1, best_thr = f1, thr_
        prob_shift = 0.5 - float(best_thr)

    # build test windows
    meta_te, X_te = build_windows_for_test(feat_te, feature_cols)
    meta_te["key"] = make_key(meta_te["station_name"], meta_te["hist_start"], meta_te["future_start"])

    # index keys
    index = index.copy()
    index["hist_start"] = pd.to_datetime(index["hist_start"])
    index["future_start"] = pd.to_datetime(index["future_start"])
    index["key"] = make_key(index["station_name"], index["hist_start"], index["future_start"])

    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba(X_te)[:, 1]
    else:
        raw = clf.predict(X_te).astype(float)
        probs = 1.0 / (1.0 + np.exp(-raw))

    # apply calibrated shift
    if prob_shift != 0.0:
        probs = np.clip(probs + prob_shift, 0.0, 1.0)

    pred_df = pd.DataFrame({"key": meta_te["key"], "y_prob": probs})

    out = index.merge(pred_df, on="key", how="left")[["id", "y_prob"]]
    out["y_prob"] = out["y_prob"].fillna(0.5)

    output_path = Path(args.predictions_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    print(f"Wrote {len(out)} rows to {output_path} (n_features={len(feature_cols)}, prob_shift={prob_shift:+.3f})")


if __name__ == "__main__":
    main()
