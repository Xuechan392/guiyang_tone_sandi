#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:23:58 2025

@author: xuechandai
"""

import os
import pandas as pd
import numpy as np

os.chdir("/Users/xuechandai/Desktop/guiyang_tone_sandi")

# Read the original labeled file
df = pd.read_csv("data/processed/f0_with_T_values_labeled.csv")

# If base_label / index do not exist, derive them from 'syllable'
if "base_label" not in df.columns or "index" not in df.columns:
    df["base_label"] = df["syllable"].astype(str).str.replace(r"\d+", "", regex=True)
    df["index"] = df["syllable"].astype(str).str.extract(r"(\d+)$")
    df["index"] = df["index"].astype("Int64")

# ---------------------------------------------
# 1. Identify all AA words based on base_label
#    A1 = index 1; A2 = index 2
# ---------------------------------------------

print("\n=========== FULL AA SANDHI ANALYSIS ===========\n")

AA_labels = df[df["index"].isin([1, 2])]["base_label"].unique()

results = []

for lbl in AA_labels:
    sub = df[(df["base_label"] == lbl) & (df["index"].isin([1, 2]))]

    if sub.empty:
        continue

    A1 = sub[sub["index"] == 1]
    A2 = sub[sub["index"] == 2]

    if A1.empty or A2.empty:
        continue

    tone_A1 = A1["tone_5deg"].value_counts().idxmax()
    tone_A2 = A2["tone_5deg"].value_counts().idxmax()

    results.append({
        "word": lbl + lbl,
        "base_label": lbl,
        "A1_tone": tone_A1,
        "A2_tone": tone_A2,
        "sandhi_pattern": f"{tone_A1}→{tone_A2}"
    })

    print(f"{lbl}{lbl}:  A1={tone_A1},  A2={tone_A2},  pattern={tone_A1}→{tone_A2}")

out = pd.DataFrame(results)
out.to_csv("data/processed/AA_sandhi_all_words.csv", index=False, encoding="utf-8-sig")
print("\nSaved AA sandhi patterns to data/processed/AA_sandhi_all_words.csv")


# --------------------------------------------------------
# 2. Detect words with multiple meanings (index > 2)
# --------------------------------------------------------

print("\n=========== MEANING-CONDITIONAL SANDHI CHECK ===========\n")

special = df[df["index"] > 2]["base_label"].unique()

for lbl in special:
    sub = df[df["base_label"] == lbl]

    print(f"\n>> Meaning contrast detected for {lbl}:")
    print(sub[["syllable", "index", "tone_5deg", "T_start", "T_end"]])

    print("\nTone distribution by meaning:")
    print(sub.groupby("index")["tone_5deg"].value_counts())
