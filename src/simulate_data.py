#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:09:39 2025

@author: xuechandai
"""

"""
simulate_data.py

Generate synthetic tone sandhi datasets inspired by Guiyang Mandarin.
This script will be responsible for:
- Defining an abstract tone inventory (e.g., 55, 35, 31, 21, 13).
- Defining a small set of "true" sandhi rules.
- Sampling random tone sequences.
- Applying the rules to create output tones.
- Injecting optional noise and missing values.
- Saving the resulting datasets to the data/ directory.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(seed=5430)


def generate_base_sequences(n_samples: int = 10000):
    """
    Generate base tone pairs (tone1, tone2) without sandhi applied yet.
    """
    # Placeholder tone inventory (you can refine later)
    tones = [55, 35, 31, 21, 13]
    tone1 = RNG.choice(tones, size=n_samples)
    tone2 = RNG.choice(tones, size=n_samples)
    df = pd.DataFrame({"tone1": tone1, "tone2": tone2})
    return df


def apply_sandhi_rules(df: pd.DataFrame):
    """
    Apply a simple set of hypothetical sandhi rules to create an output tone.
    This is where you encode your 'ground-truth' system.
    """
    def rule_row(row):
        t1, t2 = row["tone1"], row["tone2"]
        # TODO: refine these rules to match your phonology project
        if t1 == 55 and t2 == 31:
            return 21
        elif t1 == 35 and t2 == 31:
            return 33
        else:
            return t1  # default: no sandhi

    df = df.copy()
    df["tone_out"] = df.apply(rule_row, axis=1)
    return df


def inject_noise(df: pd.DataFrame, noise_prob: float = 0.05):
    """
    Randomly flip the output tone with given probability to simulate irregularity / annotation error.
    """
    df = df.copy()
    mask = RNG.random(len(df)) < noise_prob
    possible_tones = sorted(df["tone_out"].unique())
    for idx in df[mask].index:
        current = df.loc[idx, "tone_out"]
        alt_choices = [t for t in possible_tones if t != current]
        df.loc[idx, "tone_out"] = RNG.choice(alt_choices)
    return df


def inject_missing(df: pd.DataFrame, missing_prob: float = 0.1):
    """
    Randomly drop some tone_out labels to simulate missing data.
    """
    df = df.copy()
    mask = RNG.random(len(df)) < missing_prob
    df.loc[mask, "tone_out"] = np.nan
    return df


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    base = generate_base_sequences(n_samples=10000)
    clean = apply_sandhi_rules(base)
    noisy = inject_noise(clean, noise_prob=0.05)
    missing = inject_missing(clean, missing_prob=0.1)

    clean.to_csv(data_dir / "simulated_data_clean.csv", index=False)
    noisy.to_csv(data_dir / "simulated_data_noisy.csv", index=False)
    missing.to_csv(data_dir / "simulated_data_missing.csv", index=False)
    print("Datasets saved to data/ directory.")


if __name__ == "__main__":
    main()
