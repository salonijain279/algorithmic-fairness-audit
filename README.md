# Algorithmic Fairness Audit: Recidivism Risk Scoring

**Python, SQL (DuckDB), scikit-learn, Fairlearn, Streamlit**

## Problem

Automated risk-scoring tools increasingly inform high-stakes decisions — parole,
lending, hiring, insurance pricing. A model can look accurate in aggregate while
distributing its errors very differently across demographic groups. Measuring that
gap, establishing whether it's statistically real, and testing whether a concrete
mitigation actually fixes it (and at what cost) is what a fairness audit does.

## Solution

An end-to-end fairness audit built around the [COMPAS recidivism risk-scoring
dataset](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)
(ProPublica, Broward County, FL) — the dataset that anchors most of the modern
algorithmic-fairness literature. The audit:

1. Quantifies error-rate disparity (false positive / false negative rates) by race,
   with a formal statistical significance test.
2. Checks three distinct, individually well-defined fairness criteria —
   **predictive parity**, **error-rate (equalized odds) parity**, and
   **calibration** — and shows that a system can satisfy some while failing others,
   which is a mathematical property of the data, not a modeling oversight.
3. Trains a race-blind classifier and demonstrates that excluding a protected
   attribute does **not**, by itself, remove disparate outcomes.
4. Applies [Fairlearn](https://fairlearn.org/)'s `ThresholdOptimizer` post-processing
   method and measures the actual before/after reduction in the error-rate gap,
   along with the accuracy trade-off that mitigation carries.

## Architecture

```
Raw risk-score data (COMPAS)
        |
   SQL / DuckDB exploratory analysis (bias_analysis.sql)
        |
   Group confusion-matrix metrics + significance testing (src/audit.py)
        |
   Race-blind model + Fairlearn post-processing mitigation (src/mitigation.py)
        |
   Notebooks: EDA -> Audit -> Mitigation
        |
   Streamlit audit dashboard (dashboard/app.py)
```

## Repo structure

```
algorithmic-fairness-audit/
├── data/raw/compas_scores.csv          COMPAS recidivism risk-score data
├── notebooks/
│   ├── 01_eda.ipynb                    demographics, base rates, score distributions
│   ├── 02_bias_audit.ipynb             error-rate parity, predictive parity, calibration
│   └── 03_mitigation.ipynb             race-blind model + Fairlearn mitigation
├── src/
│   ├── audit.py                        group metrics, disparate impact, significance tests
│   ├── mitigation.py                   race-blind model + equalized-odds post-processing
│   └── run_audit.py                    runs the full pipeline, writes outputs/
├── dashboard/app.py                    3-page Streamlit audit console
├── outputs/                            metrics CSVs, plots, audit_summary.json
├── bias_analysis.sql
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python src/run_audit.py          # runs the full audit + mitigation pipeline
streamlit run dashboard/app.py   # interactive audit console
```

SQL analysis: `duckdb -c ".read bias_analysis.sql"`.

## Key findings (this run, n=5,278 defendants, African-American and Caucasian)

| Metric | African-American | Caucasian |
|---|---|---|
| Actual 2-year recidivism rate | 52.3% | 39.1% |
| Predicted high-risk rate | 57.6% | 33.1% |
| **False positive rate** | **42.3%** | **22.0%** |
| False negative rate | 28.5% | 49.6% |
| Precision (PPV) | 65.0% | 59.5% |

- The false-positive-rate gap (42.3% vs. 22.0%) is statistically significant
  (z=11.38, p<0.0001) — independently reproducing ProPublica's original 2016
  finding to within a couple percentage points.
- Precision (PPV) is comparable across groups — the basis of the score vendor's
  fairness defense at the time. **Both findings are simultaneously true**, which
  is the well-documented mathematical impossibility result in the fairness
  literature (Chouldechova 2017; Kleinberg, Mullainathan & Raghavan 2016): you
  generally cannot satisfy predictive parity and error-rate parity at once when
  base rates differ between groups.
- A race-blind logistic regression (race excluded entirely from training) still
  shows a real equalized-odds gap (28.4%), because `priors_count` and other
  features carry race-correlated signal — removing the protected attribute does
  not remove the disparity.
- Fairlearn's `ThresholdOptimizer` (equalized-odds constraint) reduced that gap by
  **72%** (28.4% → 7.8%), at a small accuracy cost — demonstrating a concrete,
  measurable mitigation path, not just a diagnosis.

## Limitations

- This dataset covers Broward County, FL defendants from 2013–2014; findings are
  specific to that population, time period, and the COMPAS tool as deployed then —
  they should not be read as a general claim about all risk-assessment tools.
- The provided dataset is limited to `race`, `sex`, `age_cat`, `c_charge_degree`,
  `priors_count`, `decile_score`, `score_text`, and `two_year_recid` — no
  socioeconomic, geographic, or additional case-context features are available,
  which limits what a more complete audit could examine (e.g. intersectional
  subgroups, proxy-variable analysis beyond `priors_count`).
- Only the two largest race categories (African-American, Caucasian) are analyzed
  here for group-metric stability; smaller subgroups in the full dataset would need
  larger samples for statistically reliable rates.
- Mitigation results depend on the constraint and objective chosen
  (`equalized_odds` / `accuracy_score` here); other Fairlearn configurations, or a
  different mitigation library entirely, would trade off differently.
- This is a technical audit, not a legal or policy determination — no claim is
  made here about the tool's compliance with any specific jurisdiction's law.
