#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 12:13:35 2025

@author: xuechandai
"""

"""
Extract F0 statistics from Sound + TextGrid files and convert them
to T-values (Shí Fēng normalization method) for tone analysis.

Directory layout (relative to this script):

guiyang_tone_sandi/
│
├── data/
│   ├── raw/
│   │   ├── audio/      <-- WAV files (participant01.wav, etc.)
│   │   └── textgrid/   <-- TextGrid files (participant01.TextGrid, etc.)
│   └── processed/
│       └── f0_csv/     <-- output CSV will be written here
│
└── src/
    └── extract_f0_from_textgrid.py  <-- this script
"""

import os
import glob
import math

import numpy as np
import pandas as pd
import parselmouth
from textgrid import TextGrid


# Project root = one level above this script's directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "audio")
TEXTGRID_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "textgrid")
OUTPUT_F0_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "f0_with_T_values.csv")

# Name of the tier that contains the syllable intervals
TIER_NAME = "syllable"

# F0 extraction parameters (tuned for a young adult female speaker)
PITCH_FLOOR = 60.0      # Hz
PITCH_CEILING = 450.0   # Hz
PITCH_TIME_STEP = 0.005  # seconds (5 ms, fairly dense sampling)

# ======================================================


def get_tier(textgrid: TextGrid, tier_name: str):
    """Return the tier with the given name from a TextGrid."""
    for tier in textgrid.tiers:
        if tier.name == tier_name:
            return tier
    raise ValueError(f"Tier '{tier_name}' not found in the TextGrid.")


def get_interval_pitch_stats(pitch: parselmouth.Pitch,
                             t_start: float,
                             t_end: float):
    """
    Compute F0 statistics for a given syllable interval in a robust way.

    - f0_mean: median F0 across the entire voiced portion of the interval
               (after trimming extreme values).
    - f0_min / f0_max: min / max of the voiced frames in the interval.
    - f0_start: median F0 over the FIRST third of the interval.
    - f0_end:   median F0 over the LAST third of the interval.

    This is more stable than taking a single F0 value exactly at the
    boundary times, and should better reflect rising vs. falling contours.
    """
    xs = pitch.xs()  # time stamps of pitch frames
    ys = pitch.selected_array["frequency"]  # F0 values (Hz)

    # All voiced frames within the interval
    mask_all = (xs >= t_start) & (xs <= t_end)
    vals_all = ys[mask_all]
    vals_all = vals_all[vals_all > 0]  # voiced only

    if vals_all.size == 0:
        # Entire interval unvoiced
        return {
            "f0_mean": math.nan,
            "f0_min": math.nan,
            "f0_max": math.nan,
            "f0_start": math.nan,
            "f0_end": math.nan,
        }

    # Robust central tendency: trim extremes, then use median
    q25, q75 = np.percentile(vals_all, [25, 75])
    stable_vals = vals_all[(vals_all >= q25) & (vals_all <= q75)]
    if stable_vals.size == 0:
        stable_vals = vals_all

    f0_mean = float(np.median(stable_vals))
    f0_min = float(np.min(vals_all))
    f0_max = float(np.max(vals_all))

    # Now compute start / end F0 using the first / last third
    dur = t_end - t_start
    if dur <= 0:
        return {
            "f0_mean": f0_mean,
            "f0_min": f0_min,
            "f0_max": f0_max,
            "f0_start": math.nan,
            "f0_end": math.nan,
        }

    t_first_third_end = t_start + dur / 3.0
    t_last_third_start = t_start + 2.0 * dur / 3.0

    mask_start = (xs >= t_start) & (xs <= t_first_third_end)
    mask_end = (xs >= t_last_third_start) & (xs <= t_end)

    vals_start = ys[mask_start]
    vals_end = ys[mask_end]

    vals_start = vals_start[vals_start > 0]
    vals_end = vals_end[vals_end > 0]

    f0_start = float(np.median(vals_start)) if vals_start.size > 0 else math.nan
    f0_end = float(np.median(vals_end)) if vals_end.size > 0 else math.nan

    return {
        "f0_mean": f0_mean,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_start": f0_start,
        "f0_end": f0_end,
    }


def process_one_pair(audio_path: str, textgrid_path: str):
    """
    Process one WAV + TextGrid pair and return a list of dictionaries,
    one dictionary per labeled interval in the tier.
    """
    print(f"\nProcessing: {os.path.basename(audio_path)}")

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    speaker_id = basename  # can be treated as participant ID

    sound = parselmouth.Sound(audio_path)
    tg = TextGrid.fromFile(textgrid_path)
    tier = get_tier(tg, TIER_NAME)

    # Compute pitch object for the entire sound
    pitch = sound.to_pitch(
        time_step=PITCH_TIME_STEP,
        pitch_floor=PITCH_FLOOR,
        pitch_ceiling=PITCH_CEILING,
    )

    rows = []

    for interval in tier.intervals:
        label = interval.mark.strip()
        if not label:
            continue  # skip empty labels

        t_start = float(interval.minTime)
        t_end = float(interval.maxTime)

        stats = get_interval_pitch_stats(pitch, t_start, t_end)

        row = {
            "speaker": speaker_id,
            "syllable": label,
            "t_start": t_start,
            "t_end": t_end,
            **stats,
        }
        rows.append(row)

    return rows


def compute_T_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert f0_mean / f0_start / f0_end into T-values using
    the Shí Fēng normalization method:

        T = 5 * (log10(x) - log10(b)) / (log10(a) - log10(b))

    where:
        a = global maximum F0 across all tokens (upper register)
        b = global minimum F0 across all tokens (lower register)
        x = F0 at a given measurement point (mean / start / end)
    """
    all_f0 = df["f0_mean"].replace(0, np.nan).dropna()
    if all_f0.empty:
        print("⚠ No valid f0_mean values found. Cannot compute T-values.")
        return df

    a = all_f0.max()
    b = all_f0.min()

    print(f"\nUpper pitch register (a) = {a:.2f} Hz")
    print(f"Lower pitch register (b) = {b:.2f} Hz")

    def t_of(x):
        if pd.isna(x) or x <= 0:
            return math.nan
        return 5 * (math.log10(x) - math.log10(b)) / (math.log10(a) - math.log10(b))

    for col in ["f0_mean", "f0_start", "f0_end"]:
        T_col = "T_" + col.split("_")[1]  # mean -> T_mean, start -> T_start, etc.
        df[T_col] = df[col].apply(t_of)

    return df


def main():
    os.makedirs(os.path.dirname(OUTPUT_F0_CSV), exist_ok=True)

    all_rows = []

    tg_files = glob.glob(os.path.join(TEXTGRID_DIR, "*.TextGrid"))
    if not tg_files:
        print(f"No TextGrid files found in: {TEXTGRID_DIR}")
        return

    for tg_path in tg_files:
        base = os.path.splitext(os.path.basename(tg_path))[0]
        audio_path = os.path.join(AUDIO_DIR, base + ".wav")

        if not os.path.exists(audio_path):
            print(f"⚠ Corresponding audio file not found: {audio_path}")
            continue

        rows = process_one_pair(audio_path, tg_path)
        all_rows.extend(rows)

    if not all_rows:
        print("No intervals found across any TextGrid. Nothing to export.")
        return

    df = pd.DataFrame(all_rows)

    # Compute T-values
    df = compute_T_values(df)

    # Save CSV
    df.to_csv(OUTPUT_F0_CSV, index=False, encoding="utf-8-sig")

    print(f"\n✅ Done! F0 and T-values exported to:\n{OUTPUT_F0_CSV}")
    print(f"Total intervals processed: {len(df)}")


if __name__ == "__main__":
    main()
