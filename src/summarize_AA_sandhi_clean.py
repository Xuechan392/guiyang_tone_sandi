#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:10:06 2025

@author: xuechandai
"""

import os
import pandas as pd

os.chdir("/Users/xuechandai/Desktop/guiyang_tone_sandi")

# Load enriched data
df = pd.read_csv("data/processed/kinship_tones_with_sandhi_info.csv")

# Define kinship characters only (AA set)
KINSHIP = ["爸","妈","姐","妹","哥","弟","爷","奶","公","姑","叔","婆","祖","舅","伯"]

# Keep only AA positions AND kinship characters
AA = df[
    df["base_label"].isin(KINSHIP) & 
    df["index"].isin([1,2])
].copy()

print("\n=== Clean AA Sandhi Dataset ===")
print(AA.head())

# 1. Per-character AA pattern
summary_char = (
    AA.groupby(["base_label","index"])[["citation_tone","surface_tone"]]
      .agg(lambda x: x.value_counts().index[0])
      .reset_index()
)
print("\n=== AA Sandhi Summary by Character & Position ===")
print(summary_char)

summary_char.to_csv("data/processed/AA_sandhi_summary_char.csv", index=False, encoding="utf-8-sig")

# 2. Global AA sandhi pattern (tone category × position)
summary_global = (
    AA.groupby(["citation_tone","index","surface_tone"])
      .size()
      .reset_index(name="count")
)

print("\n=== Global AA Sandhi Pattern (Counts) ===")
print(summary_global)

summary_global.to_csv("data/processed/AA_sandhi_summary_global.csv", index=False, encoding="utf-8-sig")
