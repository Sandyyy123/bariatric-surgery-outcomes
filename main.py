#!/usr/bin/env python3
"""
Bariatric Surgery Outcomes - Longitudinal Analysis Pipeline
Dr. Sandeep Grover | PhD Data Science
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings
warnings.filterwarnings("ignore")


def generate_demo_data(n=500, seed=42):
    """Generate synthetic bariatric surgery cohort for demo purposes."""
    rng = np.random.RandomState(seed)
    surgery_types = ["RYGB", "Sleeve Gastrectomy", "LAGB"]
    sexes = ["Male", "Female"]

    records = []
    for pid in range(n):
        surgery = rng.choice(surgery_types, p=[0.5, 0.35, 0.15])
        sex = rng.choice(sexes)
        baseline_weight = rng.normal(130, 20)
        baseline_bf = rng.normal(42, 8) if sex == "Female" else rng.normal(35, 7)

        # Surgery-specific weight loss patterns
        loss_rate = {"RYGB": 0.32, "Sleeve Gastrectomy": 0.26, "LAGB": 0.18}[surgery]
        regain_rate = rng.uniform(0.02, 0.08)

        weights, bodyfats, ltfu = [], [], 0
        for t_idx, t in enumerate([3, 6, 12, 24]):
            if t <= 12:
                w = baseline_weight * (1 - loss_rate * (t / 12))
            else:
                w = baseline_weight * (1 - loss_rate + regain_rate)
            w += rng.normal(0, 3)
            bf = baseline_bf * (w / baseline_weight) + rng.normal(0, 1.5)
            weights.append(max(w, 50))
            bodyfats.append(max(bf, 10))

            # LTFU probability increases over time
            ltfu_prob = 0.04 * t_idx
            if rng.rand() < ltfu_prob and ltfu == 0:
                ltfu = t
                break

        row = {
            "patient_id": pid,
            "surgery_type": surgery,
            "sex": sex,
            "baseline_weight": baseline_weight,
            "baseline_bodyfat": baseline_bf,
            "ltfu_time": ltfu if ltfu > 0 else 24,
            "ltfu_event": 1 if ltfu > 0 else 0
        }
        for i, t in enumerate([3, 6, 12, 24]):
            if i < len(weights):
                row[f"weight_t{t}"] = weights[i]
                row[f"bodyfat_t{t}"] = bodyfats[i]
            else:
                row[f"weight_t{t}"] = np.nan
                row[f"bodyfat_t{t}"] = np.nan
        records.append(row)

    return pd.DataFrame(records)


def reshape_long(df):
    """Reshape wide -> long for mixed-effects modelling."""
    timepoints = [3, 6, 12, 24]
    rows = []
    for _, r in df.iterrows():
        for t in timepoints:
            if not np.isnan(r.get(f"weight_t{t}", np.nan)):
                rows.append({
                    "patient_id": r["patient_id"],
                    "surgery_type": r["surgery_type"],
                    "sex": r["sex"],
                    "time": t,
                    "weight": r[f"weight_t{t}"],
                    "bodyfat": r.get(f"bodyfat_t{t}", np.nan),
                    "baseline_weight": r["baseline_weight"]
                })
    return pd.DataFrame(rows)


def run_lme_analysis(df_long):
    """Fit linear mixed-effects model: weight ~ time * surgery_type * sex."""
    print("\n--- Linear Mixed-Effects Model ---")
    df_model = df_long.copy()
    df_model["surgery_RYGB"] = (df_model["surgery_type"] == "RYGB").astype(int)
    df_model["surgery_SG"] = (df_model["surgery_type"] == "Sleeve Gastrectomy").astype(int)
    df_model["sex_female"] = (df_model["sex"] == "Female").astype(int)
    df_model["time_surgery_RYGB"] = df_model["time"] * df_model["surgery_RYGB"]
    df_model["time_sex"] = df_model["time"] * df_model["sex_female"]

    md = MixedLM(
        endog=df_model["weight"],
        exog=df_model[["time", "surgery_RYGB", "surgery_SG", "sex_female",
                        "time_surgery_RYGB", "time_sex"]],
        groups=df_model["patient_id"]
    )
    result = md.fit(method="lbfgs", disp=False)
    print(result.summary().tables[1])
    return result


def ltfu_analysis(df):
    """Kaplan-Meier LTFU curves by surgery type."""
    try:
        from lifelines import KaplanMeierFitter
        fig, ax = plt.subplots(figsize=(9, 5))
        colors = {"RYGB": "#6c5ce7", "Sleeve Gastrectomy": "#0984e3", "LAGB": "#fd9644"}
        for stype, grp in df.groupby("surgery_type"):
            kmf = KaplanMeierFitter()
            kmf.fit(grp["ltfu_time"], event_observed=grp["ltfu_event"], label=stype)
            kmf.plot_survival_function(ax=ax, color=colors.get(stype, "gray"), ci_show=True)
        ax.set_title("Retention (1 - LTFU) by Surgery Type", fontsize=14, fontweight="bold")
        ax.set_xlabel("Months Post-Surgery"); ax.set_ylabel("Retention Probability")
        ax.set_facecolor("#12121a"); fig.patch.set_facecolor("#0a0a0f")
        ax.tick_params(colors="#e0e0f0"); ax.xaxis.label.set_color("#888899")
        ax.yaxis.label.set_color("#888899"); ax.title.set_color("#e0e0f0")
        ax.spines["bottom"].set_color("#2a2a3a"); ax.spines["left"].set_color("#2a2a3a")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout(); plt.savefig("ltfu_kaplan_meier.png", dpi=150)
        print("\nSaved: ltfu_kaplan_meier.png")
    except ImportError:
        print("lifelines not installed - skipping KM plot")


def plot_trajectories(df_long):
    """Spaghetti plot of weight trajectories by surgery type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.patch.set_facecolor("#0a0a0f")
    surgery_types = df_long["surgery_type"].unique()
    colors_map = {"RYGB": "#6c5ce7", "Sleeve Gastrectomy": "#0984e3", "LAGB": "#fd9644"}

    for ax, stype in zip(axes, surgery_types):
        subset = df_long[df_long["surgery_type"] == stype]
        color = colors_map.get(stype, "#888")
        for pid, grp in subset.groupby("patient_id"):
            ax.plot(grp["time"], grp["weight"], alpha=0.08, color=color, linewidth=0.8)
        mean_traj = subset.groupby("time")["weight"].mean()
        ax.plot(mean_traj.index, mean_traj.values, color=color, linewidth=3, label="Mean trajectory")
        ax.set_title(stype, color="#e0e0f0", fontweight="bold")
        ax.set_facecolor("#12121a")
        ax.tick_params(colors="#888899"); ax.set_xlabel("Months", color="#888899")
        ax.spines["bottom"].set_color("#2a2a3a"); ax.spines["left"].set_color("#2a2a3a")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Weight (kg)", color="#888899")
    fig.suptitle("Weight Trajectories by Surgery Type", color="#e0e0f0", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("weight_trajectories.png", dpi=150)
    print("Saved: weight_trajectories.png")


def main():
    parser = argparse.ArgumentParser(description="Bariatric Surgery Outcomes - Longitudinal Analysis")
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--n_patients", type=int, default=500, help="Demo: number of patients to simulate")
    args = parser.parse_args()

    if args.data:
        print(f"Loading data from: {args.data}")
        df = pd.read_csv(args.data)
    else:
        print("No data file provided. Generating synthetic demo dataset...")
        df = generate_demo_data(n=args.n_patients)

    print(f"Dataset: {len(df)} patients, {df.shape[1]} columns")
    print(f"Surgery types: {df['surgery_type'].value_counts().to_dict()}")
    print(f"Missing weight data: {df[['weight_t3','weight_t6','weight_t12','weight_t24']].isna().sum().to_dict()}")

    df_long = reshape_long(df)
    print(f"Long format: {len(df_long)} rows")

    run_lme_analysis(df_long)
    plot_trajectories(df_long)
    ltfu_analysis(df)

    print("\nAnalysis complete.")
    print("Outputs: weight_trajectories.png, ltfu_kaplan_meier.png")
    print("Next: run the full notebook for mixed-effects estimates, body fat analysis, and report generation.")


if __name__ == "__main__":
    main()
