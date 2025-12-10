#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 13:13:08 2025

@author: xuechandai
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# enable Chinese characters
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']   # macOS 常见可显示中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False

# === 0. Paths & setup ===
PROJECT_ROOT = "/Users/xuechandai/Desktop/guiyang_tone_sandi"
os.chdir(PROJECT_ROOT)

DATA_PATH = "data/processed/kinship_tones_with_sandhi_info.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# === 1. Load data & keep only AA kinship tokens (index = 1 or 2) ===
df = pd.read_csv(DATA_PATH)

KINSHIP = ["ba","ma","jie","mei","ge","di","ye","nai","gong","gu","shu","po","zu","jiu","bo",
           "爸","妈","姐","妹","哥","弟","爷","奶","公","姑","叔","婆","祖","舅","伯"]  
# Supports both English and Chinese labels if needed

AA = df[
    df["base_label"].isin(KINSHIP) &
    df["index"].isin([1, 2])
].copy()

AA["citation_tone"] = AA["citation_tone"].astype("Int64")
AA["surface_tone"] = AA["surface_tone"].astype("Int64")
AA["index"] = AA["index"].astype("Int64")

print("AA tokens retained:", len(AA))


# === 2. FIGURE 1 — Surface tone distribution by syllable position (bar plot) ===
counts = (
    AA.groupby(["index", "surface_tone"])
      .size()
      .reset_index(name="count")
)

surface_levels = [1, 2, 3, 4]
positions = [1, 2]

fig, ax = plt.subplots(figsize=(6, 4))
width = 0.35
x = np.arange(len(surface_levels))

for i, pos in enumerate(positions):
    sub = counts[counts["index"] == pos]
    sub = (
        sub.set_index("surface_tone")
           .reindex(surface_levels, fill_value=0)["count"]
           .values
    )
    ax.bar(x + (i - 0.5)*width, sub, width=width, label=f"Position {pos}")

ax.set_xticks(x)
ax.set_xticklabels(surface_levels)
ax.set_xlabel("Surface tone category (1–4)")
ax.set_ylabel("Token count")
ax.set_title("Surface tone distribution by syllable position (AA kinship)")
ax.legend()

plt.tight_layout()
fig_path1 = os.path.join(FIG_DIR, "AA_surface_tone_by_position.png")
plt.savefig(fig_path1, dpi=300)
plt.close()
print("Saved:", fig_path1)


# === 3. FIGURE 2 — Citation → Surface tone matrix (heatmap via imshow) ===
table = (
    AA.groupby(["citation_tone", "surface_tone"])
      .size()
      .reset_index(name="count")
)

tone_levels = [1, 2, 3, 4]
matrix = np.zeros((4, 4), dtype=int)

for _, row in table.iterrows():
    ct = int(row["citation_tone"])
    st = int(row["surface_tone"])
    matrix[tone_levels.index(ct), tone_levels.index(st)] = row["count"]

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(matrix, cmap="Blues")

ax.set_xticks(np.arange(len(tone_levels)))
ax.set_yticks(np.arange(len(tone_levels)))
ax.set_xticklabels(tone_levels)
ax.set_yticklabels(tone_levels)

ax.set_xlabel("Surface tone")
ax.set_ylabel("Citation tone")
ax.set_title("AA sandhi: Citation → Surface tone (counts)")

for i in range(len(tone_levels)):
    for j in range(len(tone_levels)):
        ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")

plt.tight_layout()
fig_path2 = os.path.join(FIG_DIR, "AA_sandhi_citation_to_surface_matrix.png")
plt.savefig(fig_path2, dpi=300)
plt.close()
print("Saved:", fig_path2)


# === 4. FIGURE 3 — Per-character tone comparison (citation vs surface) ===
summary = (
    AA.groupby(["base_label", "index"])[["citation_tone", "surface_tone"]]
      .agg(lambda x: x.value_counts().index[0])
      .reset_index()
)

summary["label"] = (
    summary["base_label"].astype(str)
    + "_pos"
    + summary["index"].astype(str)
)

x_labels = summary["label"].tolist()
x_pos = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(max(8, len(x_labels)*0.5), 4))

ax.plot(x_pos, summary["citation_tone"], marker="o", linestyle="--", label="Citation tone")
ax.plot(x_pos, summary["surface_tone"], marker="s", linestyle="-", label="Surface tone")

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_yticks([1, 2, 3, 4])
ax.set_xlabel("Character + position (pos1 = first syllable, pos2 = second syllable)")
ax.set_ylabel("Tone category (1–4)")
ax.set_title("Per-character AA sandhi: citation vs surface tone")
ax.legend()

plt.tight_layout()
fig_path3 = os.path.join(FIG_DIR, "AA_sandhi_per_character.png")
plt.savefig(fig_path3, dpi=300)
plt.close()
print("Saved:", fig_path3)


print("\nAll tone-sandhi figures generated in:", FIG_DIR)
