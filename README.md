> **⚠️ Proprietary — All Rights Reserved.** © 2026 Sandeep Grover. This repository is licensed to Sandeep Grover and may **not** be used, run, copied, modified, distributed, or used to train models without prior written permission. Public visibility does not grant a license. See [LICENSE](LICENSE).

---

# Bariatric Surgery Outcomes Analysis

Longitudinal analysis pipeline for bariatric surgery cohorts: weight/body fat trajectory modelling, loss-to-follow-up prediction, and clinic engagement strategy.

## Features
- Linear mixed-effects models (statsmodels MixedLM / R lme4) for weight and body fat % trajectories
- Surgery type x sex interaction effects at 4 timepoints over 2 years
- Survival analysis for loss-to-follow-up (Cox regression, Kaplan-Meier)
- Missing data handling via MICE imputation
- Early vs late responder classification
- Reproducible Python + R outputs

## Setup
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --data your_data.csv
python main.py  # runs with synthetic demo data
```

## Deliverables
- Cleaned annotated notebook (Jupyter)
- Mixed-effects trajectory plots (spaghetti + predicted means)
- LTFU risk scores per patient (Cox survival model)
- Body fat % vs weight correlation grids
- Slide-ready visualisations
- Clinic engagement recommendations report
