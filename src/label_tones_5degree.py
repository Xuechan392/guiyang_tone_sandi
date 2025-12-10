#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:16:47 2025

@author: xuechandai
"""

import os
import pandas as pd
import numpy as np


os.chdir("/Users/xuechandai/Desktop/guiyang_tone_sandi")

df = pd.read_csv("data/processed/f0_with_T_values.csv")

def clamp_T(T):
    if pd.isna(T):
        return np.nan
    return min(5.0, max(0.0, T))

def t_to_height(T):
    if pd.isna(T):
        return np.nan
    h = int(round(T))
    if h < 1: h = 1
    if h > 5: h = 5
    return h

def classify_tone(Ts, Te, Tm, level_thresh=1.0, max_step=2):

    Ts = clamp_T(Ts)
    Te = clamp_T(Te)
    Tm = clamp_T(Tm)

    if pd.isna(Ts) and pd.isna(Te) and pd.isna(Tm):
        return np.nan

    if pd.isna(Ts) or pd.isna(Te):
        if pd.isna(Tm):
            return np.nan
        h = t_to_height(Tm)
        return f"{h}{h}"

    delta = Te - Ts


    if abs(delta) < level_thresh:
        h_avg = t_to_height((Ts + Te) / 2.0)
        return f"{h_avg}{h_avg}"

    h_start = t_to_height(Ts)
    h_end = t_to_height(Te)


    if h_end - h_start > max_step:
        h_end = h_start + max_step
    if h_start - h_end > max_step:
        h_start = h_end + max_step

    return f"{h_start}{h_end}"

tones = []
for Ts, Te, Tm in zip(df["T_start"], df["T_end"], df["T_mean"]):
    tone = classify_tone(Ts, Te, Tm, level_thresh=1.0, max_step=2)
    tones.append(tone)

df["tone_5deg"] = tones

output_path = "data/processed/f0_with_T_values_labeled.csv"
df.to_csv(output_path, index=False, encoding="utf-8-sig")

print(f"Done! Labeled tones saved to: {output_path}")
