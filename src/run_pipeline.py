#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 21:15:52 2025

@author: xuechandai
"""

# === RUN EVERYTHING PIPELINE ===

print("Step 1: Extract F0 from TextGrid ...")
%run src/extract_f0_from_textgrid.py

print("\nStep 2: Convert F0 → 5-degree tone labels ...")
%run src/label_tones_5degree.py

print("\nStep 3: Summarize citation tones from single-syllable data ...")
%run src/summarize_citation_tones.py

print("\nStep 4: Derive surface tone sandhi using manual citation categories ...")
%run src/derive_sandhi_with_manual_tones.py

print("\nStep 5: Summarize AA sandhi (clean dataset) ...")
%run src/summarize_AA_sandhi_clean.py

print("\nStep 6: Build probabilistic tone sandhi model ...")
%run src/build_sandhi_model.py

print("\nStep 7: Monte Carlo simulate AA sandhi ...")
%run src/simulate_sandhi.py

print("\nStep 8: Compare empirical vs simulated tone distributions ...")
%run src/compare_sim_vs_empirical.py

print("\nStep 9: Plot all tone sandhi visualizations ...")
%run src/plot_tone_sandhi_all.py

print("\n🎉 ALL STEPS COMPLETED — Pipeline Finished Successfully!")
