"""Synthetic Project Risk Dataset Generator (rule-based)

Generates a synthetic dataset inspired by structured project risk attributes.
The targets are deterministically derived from input features via explicit rules.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def assign_risk(row: pd.Series) -> tuple[str, int, int, str]:
    # Risk type (classification)
    if row["tech_complexity"] >= 4 and row["security_threats"] >= 4:
        risk_type = "Technical"
    elif row["cost_variance"] > 200_000 or row["budget_changes"] >= 4:
        risk_type = "Financial"
    elif row["reg_impact"] >= 4 or row["legal_challenges"] >= 4:
        risk_type = "Compliance"
    else:
        risk_type = "Operational"

    # Impact (regression-like integer score)
    impact = int((row["tech_complexity"] + row["budget_changes"] + row["stakeholder_influence"]) / 3 * 2 + 1)

    # Probability (regression-like percentage)
    prob = int(20 + (row["scope_creep"] + row["schedule_changes"] + row["deviate_strategy"]) / 3 * 16)

    # Response strategy (classification)
    if impact >= 7 and prob >= 70:
        response = "Mitigate"
    elif impact <= 4 and prob <= 40:
        response = "Accept"
    elif risk_type == "Financial":
        response = "Transfer"
    else:
        response = "Avoid"

    return risk_type, impact, prob, response


def make_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)

    df = pd.DataFrame()
    df["project_id"] = np.arange(1, n + 1)
    df["industry"] = rng.choice(["Construction", "Software", "Manufacturing"], n)
    df["duration_months"] = rng.randint(3, 36, n)
    df["pct_complete"] = rng.randint(0, 101, n)
    df["initial_budget"] = rng.randint(50_000, 5_000_000, n)
    df["cost_variance"] = rng.randint(-500_000, 500_000, n)
    df["scope_creep"] = rng.randint(0, 10, n)
    df["tech_complexity"] = rng.randint(1, 6, n)
    df["oper_efficiency"] = rng.randint(1, 6, n)
    df["budget_changes"] = rng.randint(1, 6, n)
    df["reg_impact"] = rng.randint(1, 6, n)
    df["deviate_strategy"] = rng.randint(1, 6, n)
    df["market_fluct"] = rng.randint(1, 6, n)
    df["reputation_impact"] = rng.randint(1, 6, n)
    df["env_impact"] = rng.randint(1, 6, n)
    df["legal_challenges"] = rng.randint(1, 6, n)
    df["security_threats"] = rng.randint(1, 6, n)
    df["supply_disrupt"] = rng.randint(1, 6, n)
    df["hr_issues"] = rng.randint(1, 6, n)
    df["schedule_changes"] = rng.randint(1, 6, n)
    df["schedule_variance_days"] = rng.randint(-60, 61, n)
    df["stakeholder_issues"] = rng.randint(1, 6, n)
    df["stakeholder_eng"] = rng.randint(1, 4, n)
    df["stakeholder_influence"] = rng.randint(1, 4, n)
    df["external_deps"] = rng.randint(1, 4, n)
    df["economic_stable"] = rng.choice([0, 1], n)
    df["quality_issues"] = rng.choice([0, 1], n)

    df[["risk_type", "impact", "probability", "response"]] = df.apply(assign_risk, axis=1, result_type="expand")
    return df


if __name__ == "__main__":
    out = make_dataset()
    out.to_csv("../data/synthetic_project_risk_dataset.csv", index=False)
    print("Saved: ../data/synthetic_project_risk_dataset.csv")
