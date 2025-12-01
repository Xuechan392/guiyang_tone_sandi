#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:13:14 2025

@author: xuechandai
"""

"""
monte_carlo.py

Run Monte Carlo experiments:
- Repeat: simulate data -> train model -> evaluate
- Aggregate accuracy distributions to study stability.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import accuracy_score

from simulate_data import generate_base_sequences, apply_sandhi_rules
from ml_models import train_models


def monte_carlo_experiment(n_runs: int = 30, n_samples: int = 5000):
    results = []

    for i in range(n_runs):
        base = generate_base_sequences(n_samples=n_samples)
        df = apply_sandhi_rules(base)
        models = train_models(df)

        for name, (model, X_test, y_test) in models.items():
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({"run": i, "model": name, "accuracy": acc})

    return pd.DataFrame(results)


def main():
    project_root = Path(__file__).resolve().parents[1]
    figs_dir = project_root / "figures"
    figs_dir.mkdir(exist_ok=True)

    df_res = monte_carlo_experiment(n_runs=30, n_samples=5000)
    df_res.to_csv(project_root / "data" / "monte_carlo_results.csv", index=False)
    print("Saved Monte Carlo results.")

    # Simple boxplot (optional)
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_res, x="model", y="accuracy")
    plt.title("Monte Carlo Accuracy Distribution")
    plt.tight_layout()
    plt.savefig(figs_dir / "monte_carlo_boxplot.png")
    plt.close()


if __name__ == "__main__":
    main()
