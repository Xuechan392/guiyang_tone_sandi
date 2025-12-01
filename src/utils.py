#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:13:38 2025

@author: xuechandai
"""

"""
utils.py

Utility functions for reproducibility and path handling.
"""

from pathlib import Path
import numpy as np


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def make_rng(seed: int = 5430):
    return np.random.default_rng(seed=seed)
