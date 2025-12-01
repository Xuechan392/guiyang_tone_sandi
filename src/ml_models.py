#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:11:49 2025

@author: xuechandai
"""

"""
ml_models.py

Train machine learning models (e.g., logistic regression, decision tree, random forest)
to predict tone_out from (tone1, tone2) and optional features.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump


def prepare_xy(df: pd.DataFrame):
    df = df.dropna(subset=["tone_out"])
    X = df[["tone1", "tone2"]].to_numpy()
    y = df["tone_out"].to_numpy()
    return X, y


def train_models(df: pd.DataFrame):
    X, y = prepare_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=5430, stratify=y
    )

    models = {}

    log_reg = LogisticRegression(max_iter=1000, multi_class="auto")
    log_reg.fit(X_train, y_train)
    models["logistic_regression"] = (log_reg, X_test, y_test)

    tree = DecisionTreeClassifier(random_state=5430)
    tree.fit(X_train, y_train)
    models["decision_tree"] = (tree, X_test, y_test)

    rf = RandomForestClassifier(
        n_estimators=200, random_state=5430, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = (rf, X_test, y_test)

    return models


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    clean_path = data_dir / "simulated_data_clean.csv"
    df = pd.read_csv(clean_path)

    models = train_models(df)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    for name, (model, X_test, y_test) in models.items():
        dump(model, models_dir / f"{name}.joblib")
        print(f"Saved {name} model.")

    print("Training complete.")


if __name__ == "__main__":
    main()
