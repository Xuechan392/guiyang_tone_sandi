#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 11:47:46 2025

@author: xuechandai
"""

import os
import pandas as pd
import numpy as np

# 0. Change to your project root
os.chdir("/Users/xuechandai/Desktop/guiyang_tone_sandi")

# 1. Load the main labeled file
df = pd.read_csv("data/processed/f0_with_T_values_labeled.csv")

# 2. Derive base_label and index from 'syllable'
#    e.g. "字1" -> base_label="字", index=1
#         "弟3" -> base_label="弟", index=3
df["syllable"] = df["syllable"].astype(str)
df["base_label"] = df["syllable"].str.replace(r"\d+", "", regex=True)
df["index"] = df["syllable"].str.extract(r"(\d+)$")
df["index"] = df["index"].astype("Int64")  # allows NaN


# 3. Manual citation tones for kinship base characters (1–4)
citation_tones = {
    "爸": 2,
    "妈": 1,
    "姐": 3,
    "妹": 4,
    "哥": 1,
    "弟": 4,
    "爷": 2,
    "奶": 1,
    "公": 1,
    "姑": 1,
    "叔": 2,
    "婆": 2,
    "祖": 3,
    "舅": 4,
    "伯": 2,
}


# 4. Build a mapping from canonical 5-degree tones (from citation_tone_summary)
#    to 4-way tone categories, based on the four tone groups.
canonical_map = {}
try:
    cit = pd.read_csv("data/processed/citation_tone_summary.csv")
    # Expect columns: tone_group (e.g. "Tone1"), selected_tone (e.g. "55")
    for _, row in cit.iterrows():
        group_name = str(row["tone_group"])   # "Tone1", "Tone2", ...
        selected = str(row["selected_tone"])  # e.g. "55", "35", "214" etc.
        # Map Tone1 -> 1, Tone2 -> 2, ...
        try:
            tone_class = int(group_name.replace("Tone", ""))
        except Exception:
            continue
        canonical_map[selected] = tone_class
    print("Canonical tone map from citation_tone_summary:", canonical_map)
except FileNotFoundError:
    print("WARNING: citation_tone_summary.csv not found; canonical_map will be empty.")
    canonical_map = {}


# 5. Map 5-degree contours to 4-way tone categories
def contour_to_category(tone_str: str) -> int:
    """
    Map a 5-degree tone label (e.g. '22', '32', '11', '24', '33', '55', '42', '54', ...)
    to a 4-way tone category (1–4).

    Priority:
    1) Direct rules you specified:
       - 22, 32 -> 2
       - 11     -> 4
       - 24     -> 1
       - 33,55  -> 1
       - 42     -> 2
       - 54     -> 3
    2) If not covered above, check if it appears as a canonical pattern
       in citation_tone_summary.csv (canonical_map).
    3) If still unknown, raise an error so you can inspect that token.
    """
    if pd.isna(tone_str):
        raise ValueError("NaN tone_5deg encountered where a contour is expected.")

    s = str(tone_str).strip()

    # 1) Explicit rules
    if s in {"22", "32"}:
        return 2
    if s == "11":
        return 4
    if s == "24":
        return 1
    if s in {"33", "55"}:
        return 1
    if s == "42":
        return 2
    if s == "54":
        return 3
    if s == "21":        # this assignment is unsure but since i got error for this I will mannually assign it as 5
        return 5         
    if s == "34":        # this assignment is unsure but since i got error for this I will mannually assign it as 5
        return 5 

    # 2) Use canonical map from citation_tone_summary (if exists)
    if s in canonical_map:
        return canonical_map[s]

    # 3) Unknown contour -> force manual check
    raise ValueError(f"Unknown 5-degree contour '{s}' for mapping to 4-way tone category.")


# 6. Attach citation_tone and surface_tone to each row
def get_citation_tone(lbl: str):
    return citation_tones.get(lbl, np.nan)

df["citation_tone"] = df["base_label"].apply(get_citation_tone)

def safe_surface_tone(x):
    return contour_to_category(x)

df["surface_tone"] = df["tone_5deg"].apply(safe_surface_tone)

# 7. Save enriched file
out_path = "data/processed/kinship_tones_with_sandhi_info.csv"
df.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved enriched tone file with citation_tone and surface_tone:\n  {out_path}")

# 8. Quick check: AA positions (index = 1 / 2) for kinship characters
aa_df = df[df["index"].isin([1, 2]) & df["base_label"].isin(citation_tones.keys())].copy()

print("\n=== Sample sandhi patterns (majority citation vs surface tone by base_label & position) ===\n")

if aa_df.empty:
    print("No AA tokens with index 1/2 found. Check your labeling.")
else:
    grouped = (
        aa_df.groupby(["base_label", "index"])[["citation_tone", "surface_tone"]]
        .agg(lambda x: x.value_counts().index[0])  # majority value
        .reset_index()
    )
    print(grouped)
