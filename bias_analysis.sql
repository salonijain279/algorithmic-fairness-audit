-- Algorithmic fairness audit queries, run via DuckDB against
-- data/raw/compas_scores.csv (the COMPAS recidivism-risk dataset).
--
-- Run with:  duckdb -c ".read bias_analysis.sql"
-- or from Python: duckdb.sql(open('bias_analysis.sql').read())

CREATE OR REPLACE VIEW compas AS
    SELECT
        *,
        CASE WHEN decile_score >= 5 THEN 1 ELSE 0 END AS pred_high_risk
    FROM read_csv_auto('data/raw/compas_scores.csv');

-- 1. Base recidivism rate and predicted high-risk rate by race ----------------------
SELECT
    race,
    COUNT(*)                          AS n,
    ROUND(AVG(two_year_recid), 4)     AS actual_recidivism_rate,
    ROUND(AVG(pred_high_risk), 4)     AS predicted_high_risk_rate
FROM compas
GROUP BY race
ORDER BY n DESC;

-- 2. False positive rate by race (flagged high-risk among those who did NOT reoffend)
SELECT
    race,
    COUNT(*) FILTER (WHERE two_year_recid = 0)                             AS actual_non_recidivists,
    SUM(pred_high_risk) FILTER (WHERE two_year_recid = 0)                  AS false_positives,
    ROUND(1.0 * SUM(pred_high_risk) FILTER (WHERE two_year_recid = 0)
        / COUNT(*) FILTER (WHERE two_year_recid = 0), 4)                   AS false_positive_rate
FROM compas
GROUP BY race
ORDER BY false_positive_rate DESC;

-- 3. False negative rate by race (missed among those who DID reoffend) --------------
SELECT
    race,
    COUNT(*) FILTER (WHERE two_year_recid = 1)                                  AS actual_recidivists,
    SUM(1 - pred_high_risk) FILTER (WHERE two_year_recid = 1)                   AS false_negatives,
    ROUND(1.0 * SUM(1 - pred_high_risk) FILTER (WHERE two_year_recid = 1)
        / COUNT(*) FILTER (WHERE two_year_recid = 1), 4)                       AS false_negative_rate
FROM compas
GROUP BY race
ORDER BY false_negative_rate DESC;

-- 4. Predictive parity: precision of the high-risk flag, by race --------------------
SELECT
    race,
    SUM(pred_high_risk)                                                    AS flagged_high_risk,
    SUM(two_year_recid) FILTER (WHERE pred_high_risk = 1)                  AS true_positives,
    ROUND(1.0 * SUM(two_year_recid) FILTER (WHERE pred_high_risk = 1)
        / NULLIF(SUM(pred_high_risk), 0), 4)                               AS precision_ppv
FROM compas
GROUP BY race
ORDER BY precision_ppv DESC;

-- 5. Calibration: within each decile score, does actual recidivism rate match by race?
SELECT
    decile_score,
    race,
    COUNT(*)                       AS n,
    ROUND(AVG(two_year_recid), 3)  AS actual_recidivism_rate
FROM compas
WHERE race IN ('African-American', 'Caucasian')
GROUP BY decile_score, race
ORDER BY decile_score, race;

-- 6. Prior-record distribution by race (a common proxy-discrimination channel) ------
SELECT
    race,
    ROUND(AVG(priors_count), 2)               AS avg_priors,
    ROUND(MEDIAN(priors_count), 1)             AS median_priors,
    ROUND(AVG(two_year_recid), 4)              AS recidivism_rate
FROM compas
GROUP BY race
ORDER BY avg_priors DESC;

-- 7. Risk category distribution (Low/Medium/High) by race ---------------------------
SELECT
    race,
    score_text,
    COUNT(*)                                                            AS n,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY race), 1) AS pct_within_race
FROM compas
GROUP BY race, score_text
ORDER BY race, score_text;

-- 8. Sex-based comparison (a second protected attribute) -----------------------------
SELECT
    sex,
    COUNT(*)                          AS n,
    ROUND(AVG(two_year_recid), 4)     AS actual_recidivism_rate,
    ROUND(AVG(pred_high_risk), 4)     AS predicted_high_risk_rate,
    ROUND(1.0 * SUM(pred_high_risk) FILTER (WHERE two_year_recid = 0)
        / COUNT(*) FILTER (WHERE two_year_recid = 0), 4) AS false_positive_rate
FROM compas
GROUP BY sex;
