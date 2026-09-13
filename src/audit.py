"""Core fairness-audit metrics: per-group confusion matrices, error-rate
parity, predictive parity, disparate impact, and statistical significance
testing of group differences.

Terminology follows the standard fairness-ML literature (Barocas, Hardt &
Narayanan; Chouldechova 2017; Hardt, Price & Srebro 2016):

- Positive prediction = the classifier flags the individual as high-risk.
- Selection rate       = P(predicted positive)
- FPR (false positive rate) = P(predicted positive | actually negative)
- FNR (false negative rate) = P(predicted negative | actually positive)
- PPV (positive predictive value / precision) = P(actually positive | predicted positive)
- Disparate impact ratio = selection_rate(unprivileged) / selection_rate(privileged)
  (the "80% rule" from US EEOC guidance flags ratios below 0.8)
- Equalized odds gap = max(|FPR difference|, |TPR difference|) across groups
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GroupMetrics:
    group: str
    n: int
    base_rate: float          # actual positive rate (ground truth)
    selection_rate: float     # predicted positive rate
    fpr: float
    fnr: float
    tpr: float
    tnr: float
    ppv: float
    npv: float
    accuracy: float


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def group_metrics(df: pd.DataFrame, group_col: str, y_true_col: str, y_pred_col: str) -> pd.DataFrame:
    rows = []
    for grp, sub in df.groupby(group_col):
        c = confusion_counts(sub[y_true_col], sub[y_pred_col])
        tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
        n = len(sub)
        rows.append(GroupMetrics(
            group=str(grp),
            n=n,
            base_rate=sub[y_true_col].mean(),
            selection_rate=sub[y_pred_col].mean(),
            fpr=fp / (fp + tn) if (fp + tn) else np.nan,
            fnr=fn / (fn + tp) if (fn + tp) else np.nan,
            tpr=tp / (tp + fn) if (tp + fn) else np.nan,
            tnr=tn / (tn + fp) if (tn + fp) else np.nan,
            ppv=tp / (tp + fp) if (tp + fp) else np.nan,
            npv=tn / (tn + fn) if (tn + fn) else np.nan,
            accuracy=(tp + tn) / n if n else np.nan,
        ))
    return pd.DataFrame([r.__dict__ for r in rows]).sort_values("n", ascending=False).reset_index(drop=True)


def disparate_impact_ratio(metrics: pd.DataFrame, privileged_group: str) -> pd.DataFrame:
    priv_rate = metrics.loc[metrics["group"] == privileged_group, "selection_rate"].iloc[0]
    out = metrics.copy()
    out["disparate_impact_ratio"] = out["selection_rate"] / priv_rate
    out["fails_80pct_rule"] = out["disparate_impact_ratio"] < 0.8
    return out


def equalized_odds_gap(metrics: pd.DataFrame) -> dict:
    fpr_gap = metrics["fpr"].max() - metrics["fpr"].min()
    tpr_gap = metrics["tpr"].max() - metrics["tpr"].min()
    return {
        "fpr_gap": fpr_gap,
        "tpr_gap": tpr_gap,
        "equalized_odds_gap": max(fpr_gap, tpr_gap),
    }


def two_proportion_z_test(count1: int, n1: int, count2: int, n2: int) -> dict:
    """Two-sided z-test for a difference in proportions (e.g. FPR between two groups)."""
    p1, p2 = count1 / n1, count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"p1": p1, "p2": p2, "z": z, "p_value": p_value}


def fpr_gap_significance(df: pd.DataFrame, group_col: str, y_true_col: str, y_pred_col: str,
                          group_a: str, group_b: str) -> dict:
    """Statistical significance of the FPR gap between two specific groups
    (restricted to actual negatives, i.e. people who did NOT reoffend)."""
    neg = df[df[y_true_col] == 0]
    a = neg[neg[group_col] == group_a]
    b = neg[neg[group_col] == group_b]
    fp_a, n_a = int(a[y_pred_col].sum()), len(a)
    fp_b, n_b = int(b[y_pred_col].sum()), len(b)
    result = two_proportion_z_test(fp_a, n_a, fp_b, n_b)
    result.update({"group_a": group_a, "group_b": group_b, "fpr_a": result["p1"], "fpr_b": result["p2"]})
    return result


def calibration_by_group(df: pd.DataFrame, group_col: str, score_col: str, y_true_col: str) -> pd.DataFrame:
    """Within each risk-score bucket, what fraction of each group actually
    reoffended? A well-calibrated score should show similar recidivism rates
    for a given score across groups."""
    rows = []
    for (grp, score), sub in df.groupby([group_col, score_col]):
        rows.append({
            "group": grp,
            "score": score,
            "n": len(sub),
            "actual_recid_rate": sub[y_true_col].mean(),
        })
    return pd.DataFrame(rows).sort_values(["score", "group"]).reset_index(drop=True)
