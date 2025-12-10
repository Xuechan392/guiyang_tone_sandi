#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:09:15 2025

@author: xuechandai
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = "/Users/xuechandai/Desktop/guiyang_tone_sandi"
os.chdir(PROJECT_ROOT)

sim = pd.read_csv("data/processed/sandhi_simulation.csv")
emp = pd.read_csv("data/processed/kinship_tones_with_sandhi_info.csv")

# Keep AA kinship only
KINSHIP = ["爸","妈","姐","妹","哥","弟","爷","奶","公","姑","叔","婆","祖","舅","伯"]
emp = emp[(emp["base_label"].isin(KINSHIP)) & (emp["index"].isin([1,2]))].copy()

# Convert type
emp["surface_tone"] = emp["surface_tone"].astype(int)

# Count real distribution
emp_counts = emp["surface_tone"].value_counts().sort_index()
sim_counts = sim["surface"].value_counts().sort_index()

# Normalize
emp_norm = emp_counts / emp_counts.sum()
sim_norm = sim_counts / sim_counts.sum()

plt.figure(figsize=(6,4))
plt.plot(emp_norm.index, emp_norm.values, marker="o", label="Empirical")
plt.plot(sim_norm.index, sim_norm.values, marker="s", label="Simulated")

plt.xlabel("Surface tone category")
plt.ylabel("Proportion")
plt.title("Empirical vs Simulated Surface Tone Distribution")
plt.legend()
plt.tight_layout()

plt.savefig("figures/sim_vs_empirical.png", dpi=300)
plt.close()

print("Saved: figures/sim_vs_empirical.png")
