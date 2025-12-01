# Guiyang Tone Sandhi Simulation  
STAT 5430 Final Project  

## Overview  

This project uses statistical computing to **simulate** and **model** tone sandhi patterns in Guiyang Mandarin.  
The main goals are:

1. Design a synthetic tone sandhi system inspired by Guiyang Mandarin.
2. Implement a **rule-based** tone sandhi predictor.
3. Train **statistical / machine learning models** to predict tone sandhi.
4. Use **simulation and Monte Carlo experiments** to study robustness, noise sensitivity, and model stability.
5. Provide a fully reproducible codebase and a written report that explains methods, results, and limitations.

This project is designed to meet the STAT 5430 final project expectations of substantial computation, algorithmic thinking, and clear communication.   

---

## Repository Structure  

```text
guiyang-tone-simulation/
├── README.md
├── environment.yml
├── .gitignore
├── data/
│   ├── README.md
│   └── (simulated CSV files will go here)
├── src/
│   ├── simulate_data.py
│   ├── rule_based_model.py
│   ├── ml_models.py
│   ├── evaluation.py
│   ├── monte_carlo.py
│   └── utils.py
├── notebook/
│   └── analysis.ipynb
├── figures/
│   └── (generated plots)
├── report/
│   └── report_outline.md
└── docs/
    └── RESEARCH_LOG.md

## How to Reproduce (planned)    
conda env create -f environment.yml
conda activate tone-sandhi-env

python src/simulate_data.py      # generate synthetic datasets
python src/rule_based_model.py   # run rule-based predictor
python src/ml_models.py          # train ML models
python src/evaluation.py         # evaluate and generate figures
python src/monte_carlo.py        # run Monte Carlo experiments


