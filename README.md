# 💬 Guiyang Mandarin Tone Sandhi 💬


## Overview

This project analyzes tone sandhi in Guiyang Mandarin syllable reduplication, I used real world (homemade lol) corpus recordings and a statistical modeling pipeline.
The project includes:

* 🗣️ Acoustic extraction of F0 from audio
* 🎵 Conversion of pitch to the 5-degree tone scale
* 🏷️ Manual + semi-automatic tone labeling
* 📊 Construction of a tone sandhi dataset
* 🧮 Probabilistic modeling
* 📑 Monte Carlo simulation to evaluate the model

This repository contains all code, processed data, and figures used in the final report.

---

## 🗂 Project Structure

```text
guiyang_tone_sandi/
├── data/
│   ├── raw/                          # Original audio + stimuli
│   │   ├── audio/                    # Original wav files
│   │   └── stimuli/                  # Stimulus lists (txt)
│   ├── processed/                    # Cleaned data, tone labels, sandhi tables, TextGrid
│   │   ├── textgrid/                 # Praat TextGrid files
│   │   ├── f0_with_T_values.csv      # Step 2 output: F0 + T-values
│   │   ├── f0_with_T_values_labeled.csv
│   │   ├── citation_tone_summary.csv
│   │   ├── kinship_tones_with_sandhi_info.csv
│   │   ├── AA_sandhi_all_words.csv
│   │   ├── AA_sandhi_summary_char.csv
│   │   ├── AA_sandhi_summary_global.csv
│   │   ├── sandhi_prob_model.csv     # Step 8 output: P(surface | citation, position)
│   │   └── sandhi_simulation.csv     # Step 9 output: Monte Carlo samples
│   └── figures/                      # All generated plots + report figures
│       ├── AA_surface_tone_by_position.png
│       ├── AA_sandhi_citation_to_surface_matrix.png
│       ├── AA_sandhi_per_character.png
│       └── sim_vs_empirical.png
├── src/
│   ├── extract_f0_from_textgrid.py       # Step 1: F0 extraction
│   ├── label_tones_5degree.py            # Step 2: convert F0 → 5-degree tones
│   ├── summarize_citation_tones.py       # Step 3: determine citation tone values
│   ├── derive_sandhi_with_manual_tones.py# Step 4: build AA sandhi dataset
│   ├── summarize_AA_sandhi_clean.py      # Step 5: clean / summarize AA sandhi table
│   ├── analyze_AA_sandhi.py              # Step 6: exploratory analysis (statistics)
│   ├── plot_tone_sandhi_all.py           # Step 7: generate all sandhi figures
│   ├── build_sandhi_model.py             # Step 8: compute P(surface | citation, position)
│   ├── simulate_sandhi.py                # Step 9: Monte Carlo simulation
│   └── compare_sim_vs_empirical.py       # Step 10: compare simulated vs empirical result
├── report/
│   └── Guiyang_Mandarin_Tone_Sandhi_Report.pdf   # Final written report
└── README.md


---

## 📊 Data Description

The dataset includes:

### 1. Monosyllabic stimuli (tone confirmation)

* One *ma* series with same segment, different tones
* 16 additional monosyllables divided into four tone groups (T1/T2/T3/T4)
* Final citation tone values:

  * T1 = 35
  * T2 = 31
  * T3 = 44
  * T4 = 14

### 2. AA kinship reduplications (tone sandhi)

17 items such as:
*爸爸, 妈妈, 姐姐, 奶奶, 舅舅, 公公, 婆婆,* etc.

Each token contains:

* citation tone
* A1/A2 position
* surface tone
* F0 contour

---

## 🔧 How to Run the Code

✔ Step 1 — Extract F0 from TextGrid
python src/extract_f0_from_textgrid.py

Output:
data/processed/f0_with_T_values.csv

✔ Step 2 — Convert F0 → 5-degree tone labels
python src/label_tones_5degree.py

Output:
data/processed/f0_with_T_values_labeled.csv

✔ Step 3 — Determine citation tone values from monosyllables
python src/summarize_citation_tones.py

Output:
data/processed/citation_tone_summary.csv

✔ Step 4 — Build sandhi dataset using manual citation tones
python src/derive_sandhi_with_manual_tones.py

Output:
data/processed/kinship_tones_with_sandhi_info.csv

✔ Step 5 — Clean AA sandhi data

(For removing unrelated syllables such as 老 from 老婆婆)

python src/summarize_AA_sandhi_clean.py

Output:
data/processed/AA_sandhi_clean.csv

✔ Step 6 — Exploratory statistics & summary
python src/analyze_AA_sandhi.py

✔ Step 7 — Generate all plots
python src/plot_tone_sandhi_all.py

Outputs saved to:
data/figures/

Includes:

A1 vs A2 surface tone histogram

Citation → Surface heatmap

Per-character tone plot

✔ Step 8 — Build probabilistic sandhi model
python src/build_sandhi_model.py

Output:
data/processed/sandhi_prob_table.csv

✔ Step 9 — Monte Carlo simulation
python src/simulate_sandhi.py

Output:
data/processed/simulated_surface_tones.csv

✔ Step 10 — Compare simulation vs empirical
python src/compare_sim_vs_empirical.py


Plot saved in:
data/figures/


## 📈 Key Results

* **A1 tones stay stable**, closely matching citation tones

* **A2 tones undergo systematic lowering/neutralization**, especially from T2 → T1/T2 and T3 → T2

* A simple probabilistic model
  [
  P(\text{surface tone} \mid \text{citation tone}, \text{position})
  ]
  successfully captures major sandhi tendencies

* Monte Carlo simulation (n = 5000) produces surface-tone patterns **highly similar to real data**, validating the model

Figures in `data/figures/` include:

* A1 vs A2 tone distribution
* Citation → Surface transition matrix
* Per-character tone plots
* Simulation vs empirical data

---

## 📚 Dependencies

This project uses:

* Python 3
* pandas
* numpy
* matplotlib
* seaborn
* os
* glob
* math
* parselmouth (Praat interface)


---

## 📑 References
Xu, X. (2011). An Introduction to Phonetics and Phonology.
Duanmu, S. The Phonology of Standard Chinese.
Tonal Sandhi Patterns Across Chinese Dialects.
Li, R. & Wang, P. (1994). Guiyang Dialect Dictionary. Jiangsu Education Press.
Bei, X. (2012). “Tone patterns and vowel patterns in Mandarin.” Wuling Journal, 131–136.
Chen, D. (2013). “Phonological variation in Guiyang Mandarin.” Journal of Guizhou Normal College, 92–99.
Luo, R. (2018). “Acoustic study of tone values and tone length in Guiyang Mandarin.” Journal of Guizhou Institute of Engineering, 63–67.
Shi, F. (2002). “The vowel pattern of Beijing Mandarin.” Nankai Linguistics, 30–36.
Shi, F. (2010). “On phonological patterns.” Nankai Linguistics., 1–14.
Tu, G. (1982). “Comments on ‘The Phonetic System of Guiyang Dialect’.” Dialect, 229–233.
Tu, G. (1987). “Noun reduplication in Guiyang.” Dialect, 202–204.
Wang, P. (1981). “The phonetic system of Guiyang dialect.” Dialect, 122–130.

---


