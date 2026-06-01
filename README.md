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
