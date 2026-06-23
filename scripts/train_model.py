"""
Trains the Draw vs Not-Draw binary classifier.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --data data/raw/sample_results.csv

Loads every CSV in data/raw/ (or just the one passed via --data), builds
features with features.py, trains an XGBoost classifier, calibrates its
probabilities (so "0.4" actually means ~40% of the time), evaluates it
properly (log loss / Brier score / precision-recall on the draw class,
NOT plain accuracy - always predicting "not draw" gets ~75% accuracy and
tells you nothing), and saves the model to models/draw_model.pkl.
"""

import argparse
import glob
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, precision_recall_curve
from sklearn.model_selection import train_test_split
import xgboost as xgb

from features import load_results_csv, build_training_table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
MODEL_PATH = os.path.join(ROOT, "models", "draw_model.pkl")

FEATURE_COLS = [
    "ppg_gap", "gf_gap", "ga_gap", "combined_goal_expectation",
    "season_points_gap", "season_gd_gap", "home_ppg_recent",
    "away_ppg_recent", "h2h_draw_rate",
]


def load_all_raw_data(explicit_path=None):
    if explicit_path:
        paths = [explicit_path]
    else:
        paths = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSVs found in {RAW_DIR}. Run scripts/generate_sample_data.py "
            "or scripts/fetch_data.py first."
        )
    frames = [load_results_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)


def train(data_path=None):
    print("Loading match data...")
    df = load_all_raw_data(data_path)
    print(f"  {len(df)} matches loaded")

    print("Building features (this walks through every match chronologically)...")
    table = build_training_table(df)
    has_odds = "bookmaker_draw_prob" in table.columns and table["bookmaker_draw_prob"].notna().any()
    feature_cols = FEATURE_COLS + (["bookmaker_draw_prob"] if has_odds else [])
    table = table.dropna(subset=feature_cols)
    print(f"  {len(table)} usable rows, {table['is_draw'].mean():.1%} draw rate")

    X = table[feature_cols]
    y = table["is_draw"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training XGBoost classifier...")
    base_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric="logloss",
        random_state=42,
    )

    print("Calibrating probabilities (isotonic, 5-fold)...")
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation (held-out test set) ---")
    print(f"Log loss:     {log_loss(y_test, probs):.4f}  (lower is better)")
    print(f"Brier score:  {brier_score_loss(y_test, probs):.4f}  (lower is better)")
    print(f"Base rate:    {y_test.mean():.1%} of test matches were draws")

    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    # report precision/recall at a threshold that flags roughly the top ~20% most confident draws
    target_idx = np.argmin(np.abs(recall - 0.3))
    print(
        f"At ~30% recall: precision = {precision[target_idx]:.1%} "
        f"(vs {y_test.mean():.1%} base rate) -- this is the 'beat the base rate' number that matters"
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to a single CSV instead of loading all of data/raw/")
    args = parser.parse_args()
    train(args.data)
