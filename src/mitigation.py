"""Bias mitigation: train a race-blind recidivism model, then apply
Fairlearn's post-processing threshold optimizer to equalize error rates
across race groups, and compare before/after fairness metrics.

This demonstrates a known, important result in the fairness literature
(Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016): you cannot
generally satisfy both predictive parity (equal precision across groups)
and equalized odds (equal error rates across groups) at the same time when
base rates differ between groups. Mitigating one type of disparity can
introduce or worsen another -- which is why an audit has to report *which*
fairness definition a mitigation optimizes for, not just claim "bias fixed."
"""
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from audit import group_metrics, equalized_odds_gap, disparate_impact_ratio

RACE_BLIND_FEATURES = ["age_cat", "c_charge_degree", "sex", "priors_count"]
CATEGORICAL = ["age_cat", "c_charge_degree", "sex"]
NUMERIC = ["priors_count"]


def build_race_blind_model() -> Pipeline:
    """A logistic regression that never sees race as a feature -- included
    to demonstrate that removing a protected attribute does NOT, by itself,
    remove disparate outcomes (because other features like priors_count are
    correlated with race through structural/historical factors)."""
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ], remainder="passthrough")
    return Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def train_and_predict(df: pd.DataFrame, seed: int = 42):
    train, test = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["two_year_recid"])

    model = build_race_blind_model()
    model.fit(train[RACE_BLIND_FEATURES], train["two_year_recid"])

    test = test.copy()
    test["pred_proba"] = model.predict_proba(test[RACE_BLIND_FEATURES])[:, 1]
    test["pred_unmitigated"] = (test["pred_proba"] >= 0.5).astype(int)
    return model, train, test


def apply_equalized_odds_mitigation(model, train: pd.DataFrame, test: pd.DataFrame, seed: int = 42):
    """Fairlearn ThresholdOptimizer: picks a (possibly different) decision
    threshold per race group so that false-positive and false-negative
    rates are equalized across groups, subject to minimizing accuracy loss.
    """
    to = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        objective="accuracy_score",
        predict_method="predict_proba",
        prefit=True,
    )
    to.fit(train[RACE_BLIND_FEATURES], train["two_year_recid"], sensitive_features=train["race"])

    test = test.copy()
    test["pred_mitigated"] = to.predict(
        test[RACE_BLIND_FEATURES], sensitive_features=test["race"], random_state=seed
    )
    return to, test


def compare_before_after(test: pd.DataFrame, privileged_group: str = "Caucasian") -> dict:
    before = group_metrics(test, "race", "two_year_recid", "pred_unmitigated")
    after = group_metrics(test, "race", "two_year_recid", "pred_mitigated")

    before_di = disparate_impact_ratio(before, privileged_group)
    after_di = disparate_impact_ratio(after, privileged_group)

    return {
        "before": before,
        "after": after,
        "before_disparate_impact": before_di,
        "after_disparate_impact": after_di,
        "before_equalized_odds_gap": equalized_odds_gap(before),
        "after_equalized_odds_gap": equalized_odds_gap(after),
    }


if __name__ == "__main__":
    df = pd.read_csv("data/raw/compas_scores.csv")
    df = df[df["race"].isin(["African-American", "Caucasian"])].reset_index(drop=True)

    model, train, test = train_and_predict(df)
    _, test = apply_equalized_odds_mitigation(model, train, test)
    result = compare_before_after(test)

    print("BEFORE mitigation:\n", result["before"].to_string(index=False))
    print("\nAFTER mitigation:\n", result["after"].to_string(index=False))
    print("\nEqualized odds gap  before:", result["before_equalized_odds_gap"])
    print("Equalized odds gap  after: ", result["after_equalized_odds_gap"])
