#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:11:16 2025

@author: xuechandai
"""

import os
import pandas as pd
import numpy as np

# Optionally ensure execution from project root (modify if needed)
os.chdir("/Users/xuechandai/Desktop/guiyang_tone_sandi")

# 1. Load the dataset with 5-degree tone labels
df = pd.read_csv("data/processed/f0_with_T_values_labeled.csv")

# Ensure tone labels are treated as strings
df["tone_5deg"] = df["tone_5deg"].astype(str)

# 2. Define citation-tone groups (single-syllable citation tones)
tone_groups = {
    "Tone1": ["妈", "花", "高", "多", "天"],
    "Tone2": ["麻", "头", "牛", "人", "狼"],
    "Tone3": ["马", "你", "我", "米", "水"],
    "Tone4": ["骂", "饭", "菜", "豆", "二"],
}

def tone_change(tone_str: str) -> int:
    """
    Compute the contour magnitude of a 5-degree tone label.
    Example:
        "22" -> 0
        "13" -> 2
        "35" -> 2
    If tone_str is invalid, return 0.
    """
    if pd.isna(tone_str):
        return 0
    s = str(tone_str)
    if len(s) < 2 or not s[0].isdigit() or not s[1].isdigit():
        return 0
    return abs(int(s[0]) - int(s[1]))


results = []

for tone_name, chars in tone_groups.items():

    subset = df[df["syllable"].isin(chars)].copy()
    subset = subset.dropna(subset=["tone_5deg"])

    if subset.empty:
        print(f"{tone_name}: no tokens found for {chars}")
        continue

    # 3. Count tone label occurrences within this tone group
    counts = subset["tone_5deg"].value_counts()

    # 3a. Remove outliers: tone labels appearing only once
    if (counts > 1).any():
        kept_labels = counts[counts > 1].index
        subset = subset[subset["tone_5deg"].isin(kept_labels)]
        counts = subset["tone_5deg"].value_counts()

    # If all labels were removed as outliers, fall back to the original counts
    if counts.empty:
        counts = df[df["syllable"].isin(chars)]["tone_5deg"].value_counts()

    if counts.empty:
        print(f"{tone_name}: still empty after fallback; skipping.")
        continue

    # Determine mode(s)
    max_count = counts.max()
    candidates = list(counts[counts == max_count].index)

    # If multiple labels tie, choose the one with the largest contour magnitude
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        chosen = max(candidates, key=tone_change)

    candidates_str = [str(c) for c in candidates]

    results.append({
        "tone_group": tone_name,
        "characters": "".join(chars),
        "selected_tone": str(chosen),
        "candidate_tones": ",".join(candidates_str),
    })

    print(f"{tone_name}: selected {chosen}  (candidates: {candidates_str})")


# 5. Save summary
output_path = "data/processed/citation_tone_summary.csv"
out_df = pd.DataFrame(results)
out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"\nCitation tone summary saved to {output_path}")
