#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 14:06:35 2025

@author: xuechandai
"""

import os
import pandas as pd
import numpy as np

PROJECT_ROOT = "/Users/xuechandai/Desktop/guiyang_tone_sandi"
os.chdir(PROJECT_ROOT)

prob = pd.read_csv("data/processed/sandhi_prob_model.csv")

def sample_surface(citation, position):
    """Randomly draw a surface tone from P(surface|citation,position)."""
    subset = prob[(prob["citation_tone"] == citation) & (prob["index"] == position)]
    tones = subset["surface_tone"].values
    probs = subset["prob"].values
    return np.random.choice(tones, p=probs)

N = 5000
sim_data = []

for _ in range(N):
    citation = np.random.choice([1,2,3,4])   # random citation tone
    pos      = np.random.choice([1,2])       # A1 or A2
    surface  = sample_surface(citation, pos)
    sim_data.append((citation, pos, surface))

sim_df = pd.DataFrame(sim_data, columns=["citation","position","surface"])
sim_df.to_csv("data/processed/sandhi_simulation.csv", index=False)

print("\nSimulation complete. Saved to data/processed/sandhi_simulation.csv")
print(sim_df.head())
