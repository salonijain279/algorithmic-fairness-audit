"""Algorithmic Fairness Audit dashboard -- Streamlit.

Three pages: Fairness Overview, Calibration Explorer, and Mitigation Comparison.
Built on src/audit.py and src/mitigation.py, using the COMPAS recidivism dataset.
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit import (  # noqa: E402
    group_metrics,
    disparate_impact_ratio,
    equalized_odds_gap,
    fpr_gap_significance,
    calibration_by_group,
)
from mitigation import train_and_predict, apply_equalized_odds_mitigation, compare_before_after  # noqa: E402

DATA_PATH = ROOT / "data" / "raw" / "compas_scores.csv"

st.set_page_config(page_title="Algorithmic Fairness Audit", layout="wide")

PRIVILEGED_GROUP = "Caucasian"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["pred_high_risk"] = (df["decile_score"] >= 5).astype(int)
    return df[df["race"].isin(["African-American", "Caucasian"])].copy()


@st.cache_data
def compute_metrics(df: pd.DataFrame):
    metrics = group_metrics(df, "race", "two_year_recid", "pred_high_risk")
    di = disparate_impact_ratio(metrics, PRIVILEGED_GROUP)
    gap = equalized_odds_gap(metrics)
    sig = fpr_gap_significance(df, "race", "two_year_recid", "pred_high_risk",
                                "African-American", "Caucasian")
    cal = calibration_by_group(df, "race", "decile_score", "two_year_recid")
    return metrics, di, gap, sig, cal


@st.cache_resource
def run_mitigation(df: pd.DataFrame):
    model, train, test = train_and_predict(df)
    _, test = apply_equalized_odds_mitigation(model, train, test)
    return compare_before_after(test)


def page_overview(df, metrics, di, gap, sig):
    st.title("Fairness Overview")
    st.caption(
        "COMPAS recidivism risk scores, Broward County FL (ProPublica's public "
        "\"Machine Bias\" dataset). Prediction = decile_score >= 5 (COMPAS's own "
        "Medium/High cutoff). Ground truth = actual 2-year reoffense."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Defendants Audited", f"{len(df):,}")
    c2.metric("Equalized Odds Gap", f"{gap['equalized_odds_gap']:.1%}")
    fpr_a = metrics.loc[metrics.group == "African-American", "fpr"].iloc[0]
    fpr_b = metrics.loc[metrics.group == "Caucasian", "fpr"].iloc[0]
    c3.metric("FPR (African-American)", f"{fpr_a:.1%}")
    c4.metric("FPR (Caucasian)", f"{fpr_b:.1%}", delta=f"{(fpr_b-fpr_a):+.1%} vs. Black defendants")

    st.info(
        f"The false-positive-rate gap is statistically significant "
        f"(z={sig['z']:.2f}, p={sig['p_value']:.1e}) — not explainable by sampling noise."
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(metrics, x="group", y=["fpr", "fnr"], barmode="group",
                     title="False Positive / False Negative Rate by Race",
                     labels={"value": "rate", "variable": "metric"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(di, x="group", y="disparate_impact_ratio", color="fails_80pct_rule",
                     title="Selection-Rate Disparate Impact Ratio (vs. Caucasian)")
        fig.add_hline(y=0.8, line_dash="dash", annotation_text="80% rule threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Full group metrics")
    st.dataframe(metrics.set_index("group"), use_container_width=True)


def page_calibration(cal):
    st.title("Calibration Explorer")
    st.caption(
        "Within each COMPAS decile score, does the actual recidivism rate line up "
        "across race groups? A well-calibrated score means the same thing for everyone."
    )
    pivot = cal.pivot(index="score", columns="group", values="actual_recid_rate")
    fig = go.Figure()
    for col in pivot.columns:
        fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col], mode="lines+markers", name=col))
    fig.add_trace(go.Scatter(x=[1, 10], y=[0, 1], mode="lines", name="reference",
                              line=dict(dash="dot", color="gray")))
    fig.update_layout(xaxis_title="COMPAS decile score", yaxis_title="actual 2-year recidivism rate",
                       title="Calibration by Race")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(cal, use_container_width=True, height=400)


def page_mitigation(df):
    st.title("Mitigation Comparison")
    st.caption(
        "A race-blind logistic regression (features: age, charge degree, sex, prior "
        "count -- race excluded) with Fairlearn's ThresholdOptimizer applied to "
        "equalize false-positive/false-negative rates across race groups."
    )
    with st.spinner("Training model and applying mitigation..."):
        result = run_mitigation(df)

    before_gap = result["before_equalized_odds_gap"]["equalized_odds_gap"]
    after_gap = result["after_equalized_odds_gap"]["equalized_odds_gap"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Equalized Odds Gap (before)", f"{before_gap:.1%}")
    c2.metric("Equalized Odds Gap (after)", f"{after_gap:.1%}", delta=f"{(after_gap-before_gap):+.1%}")
    c3.metric("Relative Reduction", f"{(1 - after_gap/before_gap):.1%}")

    col1, col2 = st.columns(2)
    with col1:
        b = result["before"][["group", "fpr", "fnr"]].melt(id_vars="group")
        b["stage"] = "before"
        a = result["after"][["group", "fpr", "fnr"]].melt(id_vars="group")
        a["stage"] = "after"
        combined = pd.concat([b, a])
        fig = px.bar(combined, x="group", y="value", color="stage", barmode="group",
                     facet_col="variable", title="Error Rates: Before vs. After Mitigation")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Before")
        st.dataframe(result["before"].set_index("group")[["fpr", "fnr", "ppv", "accuracy"]])
        st.subheader("After")
        st.dataframe(result["after"].set_index("group")[["fpr", "fnr", "ppv", "accuracy"]])

    st.caption(
        "Equalizing error rates is not free -- it can trade some accuracy or "
        "predictive parity for error-rate parity. Which fairness definition to "
        "prioritize is a deployment decision, not something a library decides for you."
    )


def main():
    df = load_data()
    metrics, di, gap, sig, cal = compute_metrics(df)

    st.sidebar.title("Algorithmic Fairness Audit")
    page = st.sidebar.radio("Page", ["Fairness Overview", "Calibration Explorer", "Mitigation Comparison"])
    st.sidebar.divider()
    st.sidebar.caption(
        "Dataset: COMPAS recidivism risk scores (ProPublica, Broward County FL). "
        "Used throughout the algorithmic-fairness research literature."
    )

    if page == "Fairness Overview":
        page_overview(df, metrics, di, gap, sig)
    elif page == "Calibration Explorer":
        page_calibration(cal)
    elif page == "Mitigation Comparison":
        page_mitigation(df)


if __name__ == "__main__":
    main()
