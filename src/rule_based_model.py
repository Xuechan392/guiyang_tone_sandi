#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:11:00 2025

@author: xuechandai
"""

"""
rule_based_model.py

Implements a hand-written rule-based tone sandhi predictor.
This mirrors the 'true' rules used in simulation and serves as a baseline.
"""

import pandas as pd
from pathlib import Path


def rule_predict_row(row):
    t1, t2 = row["tone1"], row["tone2"]
    # IMPORTANT: keep this consistent with simulate_data.apply_sandhi_rules
    if t1 == 55 and t2 == 31:
        return 21
    elif t1 == 35 and t2 == 31:
        return 33
    else:
        return t1


def apply_rule_based_model(df: pd.DataFrame):
    df = df.copy()
    df["tone_pred_rule"] = df.apply(rule_predict_row, axis=1)
    return df


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    clean_path = data_dir / "simulated_data_clean.csv"
    df = pd.read_csv(clean_path)

    df = apply_rule_based_model(df)
    out_path = data_dir / "simulated_with_rule_pred.csv"
    df.to_csv(out_path, index=False)
    print(f"Rule-based predictions saved to {out_path}")


if __name__ == "__main__":
    main()
