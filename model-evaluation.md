# Model Comparison — Fraud Prevention Case (Artefact)

This document is the technical and business evaluation of the **8 models** trained on Google Colab High-RAM (`colab-cli`, XGBoost/CatBoost on a T4 GPU) over the **523,525 transactions of 2025** (`20,485` frauds — `3.91%`). It replaces an earlier version whose protocol was optimistic (see §1).

## 1. Validation Protocol (what changed and why)

The previous protocol (random stratified 80/20 + shuffled 5-fold CV) had four problems: the decision *threshold* and the champion were chosen on the test holdout itself; features' constants came from EDA over all rows; every customer appeared in both train and test; and the single chronological check could not exercise December. The current protocol (`services/ml-inference-service/train.py`) is:

1. **Rolling-origin (walk-forward) validation.** For each month `M` in Jul–Dec/2025, train on all months before `M`, test on `M` (June is a burn-in fold used only for selection). Scored test set: **263,913 transactions, 10,721 frauds**, every row strictly later than its training data.
2. **Selection without peeking.** The F1-optimal threshold and the champion for month `M` use **only out-of-time predictions of months before `M`**. Champion rule: candidates within one fold-standard-deviation of the best mean PR-AUC are tied; ties are broken by a fixed, hardware-independent order (LightGBM > XGBoost > CatBoost > HistGB > RF > LR > Ensemble). Training time is deliberately *not* used, because it changes with GPU/CPU.
3. **Constants fit on training rows only** (`fit_domain_params`): fraud-ticket median/mean/std and the value band (5th–95th percentile of fraud amounts), recomputed per fold. The 4 regimes are domain hypotheses from full-year EDA; the test confirms them (0 frauds outside the regimes in every month) but cannot prove no new regime will appear.
4. **Seasonality features + ablation** (see §4) and a separate **December stress test**.
5. **Production models are refit on all of 2025** after selection; the production threshold is the F1 optimum of the pooled out-of-time predictions.
6. Bootstrap CIs resample rows independently and ignore temporal correlation — read them as optimistic.

## 2. Consolidated Table (out-of-time, Jul–Dec/2025)

| # | Model | Monthly PR-AUC (mean ± sd) | Pooled PR-AUC [95% CI] | Nov–Dec PR-AUC | Dec-only PR-AUC | ROC-AUC | Brier | F1 @0.50 | F1 @walk-fwd thr. | p50 | `.joblib` size |
| :-: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Naive Baseline (`Dummy`) | `0.0406 ± 0.0062` | `0.0386 [0.038, 0.039]` | `0.0421` | `0.0542` | `0.4767` | `0.03898` | `0.0000` | `0.0778` | `6.01 ms` | `7.2 KB` |
| 2 | Logistic Regression | `0.6784 ± 0.0129` | `0.6732 [0.663, 0.683]` | `0.6601` | `0.6726` | `0.9829` | `0.02015` | `0.5924` | `0.6302` | `6.86 ms` | `9.0 KB` |
| 3 | Random Forest | `0.6859 ± 0.0228` | `0.6847 [0.675, 0.694]` | `0.6905` | `0.7246` | `0.9846` | `0.01936` | `0.5943` | `0.6395` | `52.26 ms` | `11.7 MB` |
| 4 | HistGradientBoosting | `0.6842 ± 0.0263` | `0.6846 [0.675, 0.694]` | `0.6936` | `0.7294` | `0.9846` | `0.01938` | `0.6021` | `0.6442` | `8.64 ms` | `578.8 KB` |
| 5 | XGBoost | `0.6859 ± 0.0246` | `0.6861 [0.676, 0.696]` | `0.6928` | `0.7277` | `0.9846` | `0.01933` | `0.6060` | `0.6444` | `7.49 ms` | `536.8 KB` |
| 6 | **LightGBM ★ (champion)** | `0.6863 ± 0.0266` | `0.6874 [0.677, 0.698]` | `0.6974` | `0.7341` | `0.9848` | `0.01926` | `0.6077` | `0.6478` | `7.80 ms` | `554.0 KB` |
| 7 | CatBoost | `0.6853 ± 0.0268` | `0.6883 [0.678, 0.698]` | `0.7005` | `0.7336` | `0.9849` | `0.01924` | `0.6040` | `0.6483` | `6.72 ms` | `300.3 KB` |
| 8 | Soft-Voting Ensemble | `0.6871 ± 0.0269` | `0.6886 [0.678, 0.699]` | `0.6984` | `0.7354` | `0.9848` | `0.01924` | `0.6053` | `0.6484` | `48.02 ms` | `1.4 MB` |

**Reading the table honestly:** the five tree models are statistically indistinguishable (differences < 0.005 PR-AUC against a month-to-month standard deviation of ≈ 0.025). Logistic Regression trails by ≈ 0.013 pooled but the paired gap (0.0087 ± 0.011 s.e.) is not significant. There is no true "winner"; the choice below is a parsimony and operations decision, not a performance claim.

## 3. Why LightGBM (and not the others)

- **Naive Baseline:** the floor — predicts the prior (`3.9%`), PR-AUC ≈ 0.04, detects nothing at 0.50.
- **Logistic Regression:** a strong baseline thanks to the hand-built regime features; lowest PR-AUC in the benchmark and the weakest in Nov–Dec (`0.6601`). Tied on paper, but not preferred given the consistent trend and the non-linear ticket-band effect.
- **Random Forest:** equivalent accuracy, but a **11.7 MB** artifact and **52 ms** p50 — ≈ 7x slower than the boosters, unsuitable for the Kafka + FastAPI hot path.
- **HistGradientBoosting / XGBoost / CatBoost:** each within noise of LightGBM. XGBoost and CatBoost are good alternatives (CatBoost has the smallest artifact, 300 KB, and top Nov–Dec PR-AUC) and remain available in `models/`.
- **Soft-Voting Ensemble:** the best pooled numbers by 0.001, at **6x the latency (48 ms)** and 2.5x the artifact size — not worth it.
- **LightGBM (champion by the tie-break rule):** ties for the top on every metric, 7.8 ms p50, 554 KB artifact, fastest practical retraining on CPU (≈ 6 s on all of 2025), and best December-only PR-AUC among single models (`0.7341`).

## 4. Seasonality

- Volume, average ticket and new-device share are flat across months. Fraud regimes R1/R2 are stable month to month (43–52%). Real seasonality exists **only in December**: R3 (e-commerce 20h–21h) appears only then (57%), R4 (22h–23h) rises from ≈ 29% to 37%, and the overall fraud rate goes from 3.7–3.9% to **5.42%**.
- 12 calendar features were added (December flag, days to Christmas, holidays and eve, vacation season, payday window, day-of-year sin/cos, R4×December and e-commerce×December). **Ablation:** pooled PR-AUC `0.6874` with vs `0.6882` without; December `0.7341` vs `0.7348` — **no measurable gain**. In every fold before December the model never sees a December row, and with one year of data a "December effect" cannot be separated from a "holiday effect".
- **Measured consequence:** the Jan–Nov model underestimates December (predicted `4.55%` vs actual `5.42%`, −0.88 pp). At threshold 0.50, December recall is ≈ 50%; at the walk-forward threshold (0.288) it is **75.7% recall, 62.2% precision, F1 0.683**. The Step-Up tier is what absorbs seasonality the model did not learn.
- **Recommendation:** keep the calendar features (free, and the final model is refit with December), monitor calibration and alert rate in Nov/Dec 2026, and re-evaluate seasonal effects with ≥ 2 years of history.

## 5. Why the Metrics Went Down: Controlled Attribution

`scripts/ablation_protocol_drop.py` (run on Colab; full tables in `deliverables/ablation_protocol_drop.md`) changes **one factor per step** with the same models. Random-split steps are averaged over 3 seeds. LightGBM:

| Step | Condition | PR-AUC | Δ | F1 @threshold | Δ |
| --- | --- | ---: | ---: | ---: | ---: |
| S1 | Legacy: legacy features, random 80/20, threshold tuned on test | `0.6939` | | `0.6548` | |
| S2 | + new features | `0.6924` | `-0.0015` | `0.6540` | `-0.0008` |
| S3 | + score only the Jul–Dec rows of that random test (month mix) | `0.7041` | `+0.0117` | `0.6633` | `+0.0093` |
| S4 | + train on the past only (walk-forward), threshold still tuned on test | `0.6874` | **`-0.0167`** | `0.6495` | **`-0.0138`** |
| S5 | + threshold from prior months only (= current) | `0.6874` | `0.0000` | `0.6478` | `-0.0017` |

Same decomposition for the other models (PR-AUC Δ): XGBoost `-0.0010 / +0.0111 / -0.0165 / 0.0000`; Logistic Regression `+0.0032 / +0.0111 / -0.0330 / 0.0000`. Total S1→S5: LightGBM `-0.0065`, XGBoost `-0.0064`, Logistic Regression `-0.0187` PR-AUC.

**Reading it:**
- **Training on the past only is the real cost: −0.017 PR-AUC for the trees, −0.033 for Logistic Regression.** The random split lets the model train on rows from the same months (and from the future), plus the same customers. This step also trains on less data than 80% (6–11 months), so the data does not separate "leakage" from "less training data".
- **A favorable month mix hides part of it (+0.011).** In the random test, the Jul–Dec rows are *easier* than the year as a whole. The headline drop of ≈ 0.006 is therefore the net of −0.017 (past-only) and +0.011 (month mix), not a small effect.
- **The feature changes are neutral** (−0.0015 to +0.003, inside the noise); legacy vs new features under walk-forward are equal (`0.6875` vs `0.6874`).
- **The threshold protocol costs only ≈ 0.002 F1** (S4→S5). Tuning on the test set was a smaller problem than I first assumed.
- **Noise:** the legacy random-split PR-AUC spreads over `0.684–0.704` across 3 seeds, so the old and new headline numbers are not significantly different on their own; the S4-vs-S3 gap (−0.017 to −0.033) is the robust result.
- Logistic Regression is hit about twice as hard as the trees by past-only training, consistent with it adapting less to month-to-month drift. That is an inference; it was not tested separately.

## 6. Conclusion and Impact on January/2026

The convergence of the strongest models at `PR-AUC ≈ 0.687` and `ROC-AUC ≈ 0.985` suggests the irreducible noise inside the 4 regimes limits accuracy with the current columns; further gains most likely come from new real-time signals (per-customer/device velocity counters over 5–15 min windows via a Feature Store), not from more complex models.

With the **LightGBM champion** deployed in the FastAPI service and connected to Kafka, **January/2026 (`44,451` transactions, no labels)**:
- **Automatic block (`p >= 0.50`):** `1,315` transactions (`2.96%`), **R$ 39,958.90** protected, expected precision ≈ 71%.
- **Step-Up / 2FA tier (`0.288 <= p < 0.50`):** `2,324` transactions (`5.23%`) in total under protection (**R$ 69,400.10**), with `40,290` transactions (`90.6%`) approved instantly in the LOW tier.
- Expected fraud volume (sum of probabilities): `1,683` (`3.79%`), in line with January/2025 (`3.73%`). This is an estimate that relies on out-of-time calibration, which held in Jul–Nov and failed in December.
