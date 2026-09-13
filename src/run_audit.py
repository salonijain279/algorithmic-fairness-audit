"""Run the full audit + mitigation pipeline end to end and write results to outputs/.

Usage: python src/run_audit.py
"""
import json
from pathlib import Path

import pandas as pd

from audit import (
    group_metrics,
    disparate_impact_ratio,
    equalized_odds_gap,
    fpr_gap_significance,
    calibration_by_group,
)
from mitigation import train_and_predict, apply_equalized_odds_mitigation, compare_before_after

DATA_PATH = Path("data/raw/compas_scores.csv")
OUTPUTS_DIR = Path("outputs")


def main():
    df = pd.read_csv(DATA_PATH)
    df["pred_high_risk"] = (df["decile_score"] >= 5).astype(int)
    df_2race = df[df["race"].isin(["African-American", "Caucasian"])].copy()

    OUTPUTS_DIR.mkdir(exist_ok=True)

    metrics = group_metrics(df_2race, "race", "two_year_recid", "pred_high_risk")
    metrics.to_csv(OUTPUTS_DIR / "group_metrics.csv", index=False)

    di = disparate_impact_ratio(metrics, privileged_group="Caucasian")
    di.to_csv(OUTPUTS_DIR / "disparate_impact.csv", index=False)

    gap = equalized_odds_gap(metrics)

    sig = fpr_gap_significance(
        df_2race, "race", "two_year_recid", "pred_high_risk",
        "African-American", "Caucasian",
    )

    cal = calibration_by_group(df_2race, "race", "decile_score", "two_year_recid")
    cal.to_csv(OUTPUTS_DIR / "calibration_by_group.csv", index=False)

    model, train, test = train_and_predict(df_2race)
    _, test = apply_equalized_odds_mitigation(model, train, test)
    mitigation_result = compare_before_after(test)
    mitigation_result["before"].to_csv(OUTPUTS_DIR / "mitigation_before.csv", index=False)
    mitigation_result["after"].to_csv(OUTPUTS_DIR / "mitigation_after.csv", index=False)

    summary = {
        "n_total": int(len(df)),
        "n_audited_two_race": int(len(df_2race)),
        "fpr_significance_test": {k: (float(v) if not isinstance(v, str) else v) for k, v in sig.items()},
        "equalized_odds_gap_raw_score": gap,
        "equalized_odds_gap_before_mitigation": mitigation_result["before_equalized_odds_gap"],
        "equalized_odds_gap_after_mitigation": mitigation_result["after_equalized_odds_gap"],
    }
    with open(OUTPUTS_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote outputs to {OUTPUTS_DIR}/")


if __name__ == "__main__":
    main()
