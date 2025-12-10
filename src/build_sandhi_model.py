#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:03:44 2025

@author: xuechandai
"""

import os
import pandas as pd

PROJECT_ROOT = "/Users/xuechandai/Desktop/guiyang_tone_sandi"
os.chdir(PROJECT_ROOT)

df = pd.read_csv("data/processed/kinship_tones_with_sandhi_info.csv")

# Keep only AA kinship tokens
KINSHIP = ["爸","妈","姐","妹","哥","弟","爷","奶","公","姑","叔","婆","祖","舅","伯"]
AA = df[(df["base_label"].isin(KINSHIP)) & (df["index"].isin([1, 2]))].copy()

AA["citation_tone"] = AA["citation_tone"].astype(int)
AA["surface_tone"]  = AA["surface_tone"].astype(int)
AA["index"]         = AA["index"].astype(int)

# ---------------------------------------------
# Probability model: P(surface | citation, position)
# ---------------------------------------------
prob_table = (
    AA.groupby(["citation_tone","index","surface_tone"])
      .size()
      .reset_index(name="count")
)

# Normalize counts into probabilities
prob_table["prob"] = prob_table.groupby(["citation_tone","index"])["count"].transform(
    lambda x: x / x.sum()
)

prob_table.to_csv("data/processed/sandhi_prob_model.csv", index=False, encoding="utf-8-sig")

print("\n=== Probabilistic tone sandhi model ===")
print(prob_table)
print("\nSaved to: data/processed/sandhi_prob_model.csv")
