#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:12:43 2025

@author: xuechandai
"""

"""
evaluation.py

Evaluate rule-based and ML models on the simulated datasets.
Produce metrics and save figures to figures/ directory.
"""

from pathlib import Path

import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_rule_based(df: pd.DataFrame):
    mask = ~df["tone_out"].isna()
    y_true = df.loc[mask, "tone_out"].to_numpy()
    y_pred = df.loc[mask, "tone_pred_rule"].to_numpy()
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def evaluate_ml_model(model, df: pd.DataFrame):
    df = df.dropna(subset=["tone_out"])
    X = df[["tone1", "tone2"]].to_numpy()
    y_true = df["tone_out"].to_numpy()
    y_pred = model.predict(X)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return acc, cm


def plot_confusion_matrix(cm, labels, title, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    figs_dir = project_root / "figures"
    figs_dir.mkdir(exist_ok=True)

    df_rule = pd.read_csv(data_dir / "simulated_with_rule_pred.csv")
    rb_acc, rb_cm = evaluate_rule_based(df_rule)
    print(f"Rule-based accuracy: {rb_acc:.4f}")

    labels = sorted(df_rule["tone_out"].dropna().unique())
    plot_confusion_matrix(
        rb_cm,
        labels=labels,
        title="Rule-based Confusion Matrix",
        out_path=figs_dir / "confusion_matrix_rule_based.png",
    )

    # Evaluate ML models
    models_dir = project_root / "models"
    for name in ["logistic_regression", "decision_tree", "random_forest"]:
        model = load(models_dir / f"{name}.joblib")
        acc, cm = evaluate_ml_model(model, df_rule)
        print(f"{name} accuracy: {acc:.4f}")
        plot_confusion_matrix(
            cm,
            labels=labels,
            title=f"{name} Confusion Matrix",
            out_path=figs_dir / f"confusion_matrix_{name}.png",
        )


if __name__ == "__main__":
    main()
