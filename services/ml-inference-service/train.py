#!/usr/bin/env python3
"""
ATF Fraud Detection Case — Model Benchmark with a leakage-free, season-aware protocol.
Designed to execute on Google Colab (--high-mem) via scripts/colab_train_and_score.sh.

Evaluation protocol (v2):
- Rolling-origin (walk-forward) validation over 2025: for each month M in Jun..Dec the models are
  trained on months < M and scored on M. Jun is a burn-in fold (used only to pick thresholds/champion);
  Jul..Dec are the scored "test" months. Every test month is strictly later than its training data.
- Threshold and champion for scored month M are chosen ONLY from out-of-time predictions of months < M
  (walk-forward), never from M's own labels. December (the only seasonal month in a one-year dataset)
  is reported as a separate stress test.
- Data-driven constants (ticket band, fraud ticket median/std) are fit on each fold's training rows only.
- Calendar / seasonality features (December, Christmas proximity, holidays, vacation season, payday
  window, day-of-year sin/cos) are ablated against a calendar-free LightGBM.
- The legacy random 80/20 protocol is kept for 3 models, labelled optimistic, to quantify the gap.
- Production models are refit on ALL of 2025 after selection; Jan/2026 is scored with the champion.
- Progress is logged with timestamps (stdout and --log-file) so it can be tailed during a Colab run.
"""
import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
USE_GPU = False  # set from --gpu; XGBoost -> device="cuda", CatBoost -> task_type="GPU" (artifacts are converted back to CPU)
TARGET_COLUMN = "flag_fraude"

RAW_INPUT_COLUMNS = [
    "datahora_transacao",
    "nome_estabelecimento",
    "tipo_estabelecimento",
    "valor_transacao",
    "nome_cliente",
    "cidade",
    "device_type",
    "is_new_device",
    "temperatura_celsius",
    "codigo_postal",
    "score_interno",
    "turno",
    "valor_categoria",
    "cidade_regiao",
]

BASE_NUMERIC_FEATURES = [
    "valor_transacao",
    "log_valor_transacao",
    "valor_dist_from_fraud_median",
    "valor_zscore_fraud_band",
    "is_fraud_amount_sweetspot",
    "is_high_value_safe",
    "is_new_device_int",
    "temperatura_celsius",
    "score_interno",
    "hour",
    "dayofweek",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "regime_1_fuel_madrugada_sp_rj",
    "regime_2_food_almoco_sul_sudeste",
    "regime_3_ecom_dez_noite",
    "regime_4_all_noite_tardia",
    "in_any_fraud_regime",
    "new_device_in_regime",
    "mobile_new_device_in_regime",
    "desktop_new_device_in_regime",
    "tablet_in_regime",
    "regime_x_log_valor",
]

# Seasonality / calendar features. All are computable at inference time from datahora_transacao alone.
CALENDAR_FEATURES = [
    "doy_sin",
    "doy_cos",
    "is_december",
    "days_to_christmas",
    "is_vacation_season",
    "is_holiday",
    "is_holiday_eve",
    "is_month_start",
    "is_month_end",
    "is_payday_window",
    "r4_x_december",
    "ecommerce_x_december",
]

ENGINEERED_NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + CALENDAR_FEATURES

# Domain constants. DEFAULT_DOMAIN_PARAMS are the legacy hard-coded values (kept as fallback);
# fit_domain_params() re-estimates them on training rows only so no holdout information leaks in.
DEFAULT_DOMAIN_PARAMS = {
    "fraud_value_median": 27.21,
    "fraud_value_mean": 30.86,
    "fraud_value_std": 16.62,
    "sweetspot_low": 12.0,
    "sweetspot_high": 55.0,
    "high_value_safe_min": 100.0,
}

# Brazilian national holidays (hard-coded to avoid a dependency). Extend before scoring later years.
HOLIDAYS = {
    date(2025, 1, 1), date(2025, 3, 3), date(2025, 3, 4), date(2025, 4, 18), date(2025, 4, 21),
    date(2025, 5, 1), date(2025, 6, 19), date(2025, 9, 7), date(2025, 10, 12), date(2025, 11, 2),
    date(2025, 11, 15), date(2025, 11, 20), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 2, 16), date(2026, 2, 17), date(2026, 4, 3), date(2026, 4, 21),
    date(2026, 5, 1), date(2026, 6, 4), date(2026, 9, 7), date(2026, 10, 12), date(2026, 11, 2),
    date(2026, 11, 15), date(2026, 11, 20), date(2026, 12, 25),
}

CATEGORICAL_FEATURES = [
    "nome_estabelecimento",
    "tipo_estabelecimento",
    "cidade",
    "device_type",
    "turno",
    "valor_categoria",
    "cidade_regiao",
]

KNOWN_CATEGORIES = [
    [
        "Clothing Store",
        "Electronics Store",
        "Gas Station",
        "Online Shop 1",
        "Online Shop 2",
        "Pharmacy Z",
        "Restaurant X",
        "Restaurant Y",
        "Supermarket A",
        "Supermarket B",
    ],
    ["E-commerce", "Food & Beverage", "Fuel", "Pharmacy", "Retail", "Supermarket"],
    [
        "Belo Horizonte",
        "Curitiba",
        "Porto Alegre",
        "Recife",
        "Rio de Janeiro",
        "Salvador",
        "São Paulo",
    ],
    ["desktop", "mobile", "tablet"],
    ["madrugada", "manha", "tarde", "noite"],
    ["baixo", "medio", "alto"],
    ["Nordeste", "Sudeste", "Sul"],
]


def fit_domain_params(train_df: pd.DataFrame) -> dict:
    """Estimate data-driven constants from TRAINING rows only (fraud ticket distribution)."""
    fraud_values = train_df.loc[train_df[TARGET_COLUMN].astype(bool), "valor_transacao"].astype(float)
    if len(fraud_values) < 50:
        return dict(DEFAULT_DOMAIN_PARAMS)
    return {
        "fraud_value_median": float(fraud_values.median()),
        "fraud_value_mean": float(fraud_values.mean()),
        "fraud_value_std": float(max(fraud_values.std(), 1e-6)),
        "sweetspot_low": float(fraud_values.quantile(0.05)),
        "sweetspot_high": float(fraud_values.quantile(0.95)),
        "high_value_safe_min": DEFAULT_DOMAIN_PARAMS["high_value_safe_min"],
    }


def load_domain_params(path: Path) -> dict:
    """Load params saved by train.py; fall back to the legacy constants if the file is absent."""
    if path.is_file():
        return {**DEFAULT_DOMAIN_PARAMS, **json.loads(path.read_text(encoding="utf-8"))}
    return dict(DEFAULT_DOMAIN_PARAMS)


def engineer_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Deterministic feature engineering from raw transaction columns.

    `params` carries the constants fit on training data (see fit_domain_params); with params=None the
    legacy defaults are used. Regime definitions are domain hypotheses taken from EDA (documented in
    the notebook); the numeric constants are NOT derived from evaluation rows.
    """
    p = {**DEFAULT_DOMAIN_PARAMS, **(params or {})}
    out = pd.DataFrame(index=df.index)
    dt = pd.to_datetime(df["datahora_transacao"], errors="coerce")
    hour = dt.dt.hour.fillna(0).astype(int)
    month = dt.dt.month.fillna(1).astype(int)
    day = dt.dt.day.fillna(1).astype(int)
    dayofweek = dt.dt.day_of_week.fillna(0).astype(int)

    valor = df["valor_transacao"].astype(float)
    log_valor = np.log1p(np.maximum(0.0, valor))
    is_new_dev = df["is_new_device"].astype(bool).astype(int)

    out["valor_transacao"] = valor
    out["log_valor_transacao"] = log_valor
    out["valor_dist_from_fraud_median"] = np.abs(log_valor - np.log1p(p["fraud_value_median"]))
    out["valor_zscore_fraud_band"] = (valor - p["fraud_value_mean"]) / p["fraud_value_std"]
    out["is_fraud_amount_sweetspot"] = ((valor >= p["sweetspot_low"]) & (valor <= p["sweetspot_high"])).astype(int)
    out["is_high_value_safe"] = (valor >= p["high_value_safe_min"]).astype(int)
    out["is_new_device_int"] = is_new_dev
    out["temperatura_celsius"] = df["temperatura_celsius"].astype(float)
    out["score_interno"] = df["score_interno"].astype(float)
    out["hour"] = hour
    out["dayofweek"] = dayofweek
    out["is_weekend"] = (dayofweek >= 5).astype(int)
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    r1 = (
        (df["tipo_estabelecimento"] == "Fuel")
        & (hour.between(0, 8))
        & (df["cidade"].isin(["São Paulo", "Rio de Janeiro"]))
    ).astype(int)
    r2 = (
        (df["tipo_estabelecimento"] == "Food & Beverage")
        & (hour.between(12, 14))
        & (df["cidade_regiao"].isin(["Sudeste", "Sul"]))
    ).astype(int)
    r3 = (
        (df["tipo_estabelecimento"] == "E-commerce")
        & (month == 12)
        & (hour.between(20, 21))
    ).astype(int)
    r4 = (hour.between(22, 23)).astype(int)
    any_regime = ((r1 == 1) | (r2 == 1) | (r3 == 1) | (r4 == 1)).astype(int)

    out["regime_1_fuel_madrugada_sp_rj"] = r1
    out["regime_2_food_almoco_sul_sudeste"] = r2
    out["regime_3_ecom_dez_noite"] = r3
    out["regime_4_all_noite_tardia"] = r4
    out["in_any_fraud_regime"] = any_regime
    out["new_device_in_regime"] = any_regime * is_new_dev
    out["mobile_new_device_in_regime"] = (
        (any_regime == 1) & (is_new_dev == 1) & (df["device_type"] == "mobile")
    ).astype(int)
    out["desktop_new_device_in_regime"] = (
        (any_regime == 1) & (is_new_dev == 1) & (df["device_type"] == "desktop")
    ).astype(int)
    out["tablet_in_regime"] = (
        (any_regime == 1) & (df["device_type"] == "tablet")
    ).astype(int)
    out["regime_x_log_valor"] = any_regime * log_valor

    # ---- Seasonality / calendar features ----
    doy = dt.dt.day_of_year.fillna(1).astype(int)
    year = dt.dt.year.fillna(2025).astype(int)
    days_in_month = dt.dt.days_in_month.fillna(30).astype(int)
    py_dates = dt.dt.date
    is_december = (month == 12).astype(int)
    # Days until Christmas Day (same year), clipped to [0, 31]; dates after Dec 25 and non-December
    # dates are treated as "far" (31) so January never looks like Christmas week.
    days_to_xmas = (pd.to_datetime(year.astype(str) + "-12-25") - dt.dt.normalize()).dt.days
    days_to_xmas = days_to_xmas.where(days_to_xmas >= 0, 31).clip(0, 31).fillna(31).astype(int)
    holiday = py_dates.map(lambda d: d in HOLIDAYS if isinstance(d, date) else False).astype(int)
    holiday_eve = py_dates.map(
        lambda d: (date.fromordinal(d.toordinal() + 1) in HOLIDAYS) if isinstance(d, date) else False
    ).astype(int)

    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 366.0)
    out["is_december"] = is_december
    out["days_to_christmas"] = days_to_xmas
    out["is_vacation_season"] = month.isin([1, 7, 12]).astype(int)
    out["is_holiday"] = holiday
    out["is_holiday_eve"] = holiday_eve
    out["is_month_start"] = (day <= 3).astype(int)
    out["is_month_end"] = ((days_in_month - day) <= 1).astype(int)
    out["is_payday_window"] = (day.between(5, 7) | day.between(20, 22)).astype(int)
    out["r4_x_december"] = r4 * is_december
    out["ecommerce_x_december"] = (df["tipo_estabelecimento"] == "E-commerce").astype(int) * is_december

    for col in CATEGORICAL_FEATURES:
        out[col] = df[col].astype(str)

    return out[ENGINEERED_NUMERIC_FEATURES + CATEGORICAL_FEATURES]


def build_preprocessor(scale_numeric: bool = False, numeric_features: list | None = None) -> ColumnTransformer:
    numeric_features = numeric_features or ENGINEERED_NUMERIC_FEATURES
    num_transformer = StandardScaler() if scale_numeric else "passthrough"
    cat_transformer = OneHotEncoder(
        categories=KNOWN_CATEGORIES,
        handle_unknown="ignore",
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_models(seed: int, n_estimators: int, n_jobs: int, numeric_features: list | None = None):
    naive_baseline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False, numeric_features=numeric_features)),
            ("classifier", DummyClassifier(strategy="prior", random_state=seed)),
        ]
    )

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True, numeric_features=numeric_features)),
            (
                "classifier",
                LogisticRegression(
                    C=0.5,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False, numeric_features=numeric_features)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=12,
                    min_samples_leaf=20,
                    max_features="sqrt",
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
        ]
    )

    hist_gb = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False, numeric_features=numeric_features)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.04,
                    max_leaf_nodes=23,
                    min_samples_leaf=60,
                    l2_regularization=2.0,
                    random_state=seed,
                ),
            ),
        ]
    )

    xgboost_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False, numeric_features=numeric_features)),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    n_estimators=n_estimators,
                    learning_rate=0.04,
                    max_depth=5,
                    min_child_weight=15,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.5,
                    reg_lambda=3.0,
                    tree_method="hist",
                    device="cuda" if USE_GPU else "cpu",
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
        ]
    )

    lightgbm_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False, numeric_features=numeric_features)),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=n_estimators,
                    learning_rate=0.04,
                    num_leaves=23,
                    max_depth=5,
                    min_child_samples=60,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_alpha=0.5,
                    reg_lambda=3.0,
                    n_jobs=n_jobs,
                    random_state=seed,
                    verbose=-1,
                ),
            ),
        ]
    )

    catboost_model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False, numeric_features=numeric_features)),
            (
                "classifier",
                CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=5,
                    l2_leaf_reg=5.0,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    random_seed=seed,
                    thread_count=n_jobs,
                    **({"task_type": "GPU"} if USE_GPU else {}),
                    verbose=0,
                ),
            ),
        ]
    )

    return {
        "naive_baseline": naive_baseline,
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "hist_gradient_boosting": hist_gb,
        "xgboost": xgboost_model,
        "lightgbm": lightgbm_model,
        "catboost": catboost_model,
    }


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if np.any(mask):
            ece += (np.sum(mask) / n) * abs(float(np.mean(y_prob[mask])) - float(np.mean(y_true[mask])))
    return float(ece)


def calculate_metrics(labels, probabilities, predictions):
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "ece": expected_calibration_error(labels, probabilities),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "false_positive_rate": float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0,
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def find_optimal_f1_threshold(labels, probabilities):
    if len(np.unique(probabilities)) <= 1:
        return 0.5, 0.0
    precisions, recalls, thresholds = precision_recall_curve(labels, probabilities)
    f1_scores = (2 * precisions[:-1] * recalls[:-1]) / np.maximum(
        precisions[:-1] + recalls[:-1], 1e-12
    )
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def bootstrap_ci(labels, probabilities, predictions, n_boot=200, seed=2026):
    rng = np.random.default_rng(seed)
    n = len(labels)
    pr_aucs, roc_aucs, f1s = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        p_b = probabilities[idx]
        pred_b = predictions[idx]
        pr_aucs.append(average_precision_score(y_b, p_b))
        roc_aucs.append(roc_auc_score(y_b, p_b))
        f1s.append(f1_score(y_b, pred_b, zero_division=0))
    return {
        "average_precision_95ci": [
            float(np.percentile(pr_aucs, 2.5)),
            float(np.percentile(pr_aucs, 97.5)),
        ],
        "roc_auc_95ci": [
            float(np.percentile(roc_aucs, 2.5)),
            float(np.percentile(roc_aucs, 97.5)),
        ],
        "f1_95ci": [
            float(np.percentile(f1s, 2.5)),
            float(np.percentile(f1s, 97.5)),
        ],
    }


def measure_inference_latency_ms(estimator, sample_df: pd.DataFrame, n_runs: int = 40):
    single_row = sample_df.iloc[[0]]
    # Warmup
    estimator.predict_proba(single_row)
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        estimator.predict_proba(single_row)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }




# ----------------------------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------------------------
_T0 = time.perf_counter()
_LOG_FILE: Path | None = None


def log(message: str) -> None:
    """Timestamped, flushed log line (stdout + optional file) so a Colab run can be tailed live."""
    line = f"[{time.strftime('%H:%M:%S')} +{time.perf_counter() - _T0:7.1f}s] {message}"
    print(line, flush=True)
    if _LOG_FILE is not None:
        with _LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


# ----------------------------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------------------------
ENSEMBLE_NAME = "soft_voting_ensemble"
ENSEMBLE_MEMBERS = ["catboost", "lightgbm", "xgboost"]
ABLATION_NAME = "lightgbm_no_calendar"
FOLD_METRICS = ["average_precision", "roc_auc", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"]


def fold_metrics(labels, probabilities) -> dict:
    preds = (probabilities >= 0.5).astype(np.int32)
    has_both = len(np.unique(labels)) > 1
    return {
        "average_precision": float(average_precision_score(labels, probabilities)) if has_both else float("nan"),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if has_both else float("nan"),
        "precision_at_0_5": float(precision_score(labels, preds, zero_division=0)),
        "recall_at_0_5": float(recall_score(labels, preds, zero_division=0)),
        "f1_at_0_5": float(f1_score(labels, preds, zero_division=0)),
    }


def summarize_folds(per_fold: list[dict]) -> dict:
    return {
        metric: {
            "mean": float(np.mean([f[metric] for f in per_fold])),
            "std": float(np.std([f[metric] for f in per_fold])),
            "folds": [float(f[metric]) for f in per_fold],
        }
        for metric in FOLD_METRICS
    }


# Deterministic, hardware-independent preference order used ONLY to break statistical ties
# (training time is NOT used: it changes with GPU/CPU and would make the champion non-reproducible).
TIE_BREAK_ORDER = [
    "lightgbm",
    "xgboost",
    "catboost",
    "hist_gradient_boosting",
    "random_forest",
    "logistic_regression",
    "soft_voting_ensemble",
]


def select_champion(ap_by_fold: dict) -> str:
    """Parsimonious champion rule using ONLY out-of-time fold scores supplied by the caller.

    The best mean PR-AUC defines the reference; every candidate within one fold-std of it is "tied", and the
    first tied candidate in TIE_BREAK_ORDER (single gradient-boosted model, small artifact, low latency) wins.
    """
    means = {m: float(np.mean(v)) for m, v in ap_by_fold.items()}
    best = max(means, key=means.get)
    tolerance = float(np.std(ap_by_fold[best]))
    tied = [m for m, v in means.items() if v >= means[best] - tolerance]
    return min(tied, key=lambda m: TIE_BREAK_ORDER.index(m) if m in TIE_BREAK_ORDER else len(TIE_BREAK_ORDER))


def concat_probs(probs_by_fold: dict, months: list[int]) -> np.ndarray:
    return np.concatenate([probs_by_fold[m] for m in months])


def make_cpu_portable(estimator) -> None:
    """GPU-trained XGBoost boosters must be switched back to CPU before serialization so the CPU-only
    inference service can load and run them (CatBoost GPU models already predict on CPU)."""
    if isinstance(estimator, VotingClassifier):
        for member in estimator.estimators_:
            make_cpu_portable(member)
    elif isinstance(estimator, Pipeline):
        make_cpu_portable(estimator.steps[-1][1])
    elif isinstance(estimator, XGBClassifier):
        estimator.set_params(device="cpu")
        estimator.get_booster().set_param({"device": "cpu"})


def parse_args():
    parser = argparse.ArgumentParser(description="Season-aware, leakage-free benchmark of fraud models; scores Jan 2026.")
    parser.add_argument("--train-data", type=Path, default=ROOT_DIR / "data" / "transacoes_historico_2025.csv")
    parser.add_argument("--jan-data", type=Path, default=ROOT_DIR / "data" / "transacoes_jan2026.csv")
    parser.add_argument("--models-dir", type=Path, default=ROOT_DIR / "models")
    parser.add_argument("--deliverables-dir", type=Path, default=ROOT_DIR / "deliverables")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--first-test-month", type=int, default=7, help="First scored month (month before it is burn-in).")
    parser.add_argument("--gpu", action="store_true", help="Train XGBoost/CatBoost on CUDA (artifacts are converted back to CPU).")
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--skip-legacy", action="store_true", help="Skip the optimistic random 80/20 comparison.")
    parser.add_argument("--max-rows", type=int, default=0, help="Debug: subsample rows (stratified by month).")
    parser.add_argument("--log-file", type=Path, default=None, help="Also append log lines to this file.")
    return parser.parse_args()


def main():
    global _LOG_FILE, USE_GPU
    args = parse_args()
    USE_GPU = bool(args.gpu)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text("", encoding="utf-8")
        _LOG_FILE = args.log_file
    if not args.train_data.is_file():
        raise FileNotFoundError(f"Training dataset not found: {args.train_data}")

    log(f"[1/8] Loading historical dataset: {args.train_data} (python {platform.python_version()}, n_jobs={args.n_jobs}, gpu={USE_GPU})")
    dataset = pd.read_csv(args.train_data)
    if args.max_rows and args.max_rows < len(dataset):
        dataset = (
            dataset.assign(_m=pd.to_datetime(dataset["datahora_transacao"]).dt.month)
            .groupby("_m", group_keys=False)
            .sample(frac=args.max_rows / len(dataset), random_state=args.seed)
            .drop(columns="_m")
            .reset_index(drop=True)
        )
        log(f"      DEBUG subsample active: {len(dataset):,} rows")
    dataset["_month"] = pd.to_datetime(dataset["datahora_transacao"]).dt.month
    labels = dataset[TARGET_COLUMN].astype(int).to_numpy(dtype=np.int32)
    months_all = dataset["_month"].to_numpy()
    log(f"      rows={len(dataset):,} fraud_rate={labels.mean():.4%} months={sorted(set(months_all.tolist()))}")

    fold_months = list(range(args.first_test_month - 1, 13))  # burn-in month + scored months
    scored_months = [m for m in fold_months if m >= args.first_test_month]
    model_names = list(build_models(args.seed, args.n_estimators, args.n_jobs).keys())
    no_cal_features = BASE_NUMERIC_FEATURES

    # ------------------------------------------------------------------------------------------
    # [2/8] Rolling-origin out-of-time predictions
    # ------------------------------------------------------------------------------------------
    log(f"[2/8] Rolling-origin validation: burn-in month {fold_months[0]}, scored months {scored_months}")
    oot_probs: dict[str, dict[int, np.ndarray]] = {}
    oot_labels: dict[int, np.ndarray] = {}
    oot_raw: dict[int, pd.DataFrame] = {}
    fit_times: dict[str, list[float]] = {}
    fold_info = []
    regime_checks = {}
    for fold_idx, month in enumerate(fold_months, start=1):
        train_mask = months_all < month
        test_mask = months_all == month
        train_raw = dataset.loc[train_mask]
        test_raw_m = dataset.loc[test_mask]
        params_m = fit_domain_params(train_raw)  # constants fit on TRAIN rows of this fold only
        x_train = engineer_features(train_raw, params_m)
        x_test = engineer_features(test_raw_m, params_m)
        y_train, y_test = labels[train_mask], labels[test_mask]
        oot_labels[month] = y_test
        oot_raw[month] = test_raw_m
        in_regime = x_test["in_any_fraud_regime"].to_numpy() == 1
        regime_checks[month] = {
            "test_rows": int(len(y_test)),
            "fraud_outside_regimes": int(y_test[~in_regime].sum()),
            "in_regime_fraud_rate": float(y_test[in_regime].mean()) if in_regime.any() else None,
            "train_params": params_m,
        }
        log(
            f"  [fold {fold_idx}/{len(fold_months)}] test=month {month} train_rows={int(train_mask.sum()):,} "
            f"test_rows={int(test_mask.sum()):,} test_fraud={y_test.mean():.3%} "
            f"fraud_outside_regimes={regime_checks[month]['fraud_outside_regimes']}"
        )
        fold_models = build_models(args.seed, args.n_estimators, args.n_jobs)
        fold_models[ABLATION_NAME] = build_models(args.seed, args.n_estimators, args.n_jobs, no_cal_features)["lightgbm"]
        for name, estimator in fold_models.items():
            t0 = time.perf_counter()
            estimator.fit(x_train, y_train)
            elapsed = time.perf_counter() - t0
            fit_times.setdefault(name, []).append(elapsed)
            probs = estimator.predict_proba(x_test)[:, 1]
            oot_probs.setdefault(name, {})[month] = probs
            ap = average_precision_score(y_test, probs) if len(np.unique(y_test)) > 1 else float("nan")
            log(f"      {name:24s} fit={elapsed:6.1f}s  PR-AUC(month {month})={ap:.4f}")
        # soft voting == mean of member probabilities (equal weights), no extra fit needed
        oot_probs.setdefault(ENSEMBLE_NAME, {})[month] = np.mean(
            [oot_probs[m][month] for m in ENSEMBLE_MEMBERS], axis=0
        )
        fit_times.setdefault(ENSEMBLE_NAME, []).append(sum(fit_times[m][-1] for m in ENSEMBLE_MEMBERS))
        fold_info.append({"month": month, "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
                          "test_fraud_rate": float(y_test.mean()), "scored": month in scored_months})

    all_names = model_names + [ENSEMBLE_NAME]
    candidates = [n for n in all_names if n != "naive_baseline"]

    # ------------------------------------------------------------------------------------------
    # [3/8] Walk-forward threshold + champion selection (never uses the scored month's own labels)
    # ------------------------------------------------------------------------------------------
    log("[3/8] Walk-forward threshold/champion selection from prior months only...")
    wf_threshold: dict[str, dict[int, float]] = {n: {} for n in all_names + [ABLATION_NAME]}
    wf_champion: dict[int, str] = {}
    for month in scored_months:
        prior = [m for m in fold_months if m < month]
        y_prior = np.concatenate([oot_labels[m] for m in prior])
        for name in wf_threshold:
            wf_threshold[name][month] = find_optimal_f1_threshold(y_prior, concat_probs(oot_probs[name], prior))[0]
        ap_prior = {
            n: [average_precision_score(oot_labels[m], oot_probs[n][m]) for m in prior] for n in candidates
        }
        wf_champion[month] = select_champion(ap_prior)
        log(f"      month {month}: champion={wf_champion[month]} champion_threshold={wf_threshold[wf_champion[month]][month]:.3f}")

    y_test_pool = np.concatenate([oot_labels[m] for m in scored_months])
    raw_test_pool = pd.concat([oot_raw[m] for m in scored_months], ignore_index=True)
    month_test_pool = np.concatenate([np.full(len(oot_labels[m]), m) for m in scored_months])

    cv_results, test_results_05, test_results_optimal = {}, {}, {}
    chrono_results, december_results, bootstrap_cis, calibration_curves = {}, {}, {}, {}
    per_month_ap: dict[str, dict[str, float]] = {}
    pooled_prod_threshold: dict[str, float] = {}
    pooled_probs_store: dict[str, np.ndarray] = {}
    for name in all_names + [ABLATION_NAME]:
        probs_pool = concat_probs(oot_probs[name], scored_months)
        pooled_probs_store[name] = probs_pool
        per_fold = [fold_metrics(oot_labels[m], oot_probs[name][m]) for m in scored_months]
        per_month_ap[name] = {str(m): per_fold[i]["average_precision"] for i, m in enumerate(scored_months)}
        cv_results[name] = summarize_folds(per_fold)
        preds_05 = (probs_pool >= 0.5).astype(np.int32)
        test_results_05[name] = calculate_metrics(y_test_pool, probs_pool, preds_05)
        wf_thr_vec = np.concatenate([np.full(len(oot_labels[m]), wf_threshold[name][m]) for m in scored_months])
        preds_opt = (probs_pool >= wf_thr_vec).astype(np.int32)
        all_months_pool = [m for m in fold_months]
        pooled_prod_threshold[name] = find_optimal_f1_threshold(
            np.concatenate([oot_labels[m] for m in all_months_pool]), concat_probs(oot_probs[name], all_months_pool)
        )[0]
        test_results_optimal[name] = {
            "threshold": pooled_prod_threshold[name],
            "walk_forward_thresholds": {str(m): wf_threshold[name][m] for m in scored_months},
            **calculate_metrics(y_test_pool, probs_pool, preds_opt),
        }
        nd = [m for m in scored_months if m >= 11]
        if nd:
            p_nd, y_nd = concat_probs(oot_probs[name], nd), np.concatenate([oot_labels[m] for m in nd])
            chrono_results[name] = calculate_metrics(y_nd, p_nd, (p_nd >= 0.5).astype(np.int32))
        if 12 in scored_months:
            p12, y12 = oot_probs[name][12], oot_labels[12]
            thr12 = wf_threshold[name][12]
            december_results[name] = {
                "threshold_walk_forward": thr12,
                "at_0_5": calculate_metrics(y12, p12, (p12 >= 0.5).astype(np.int32)),
                "at_walk_forward_threshold": calculate_metrics(y12, p12, (p12 >= thr12).astype(np.int32)),
            }
        bootstrap_cis[name] = bootstrap_ci(y_test_pool, probs_pool, preds_05, n_boot=args.n_boot, seed=args.seed)
        if name != "naive_baseline":
            frac_pos, mean_pred = calibration_curve(y_test_pool, probs_pool, n_bins=10, strategy="uniform")
            calibration_curves[name] = {
                "fraction_of_positives": [float(v) for v in frac_pos],
                "mean_predicted_value": [float(v) for v in mean_pred],
            }
        log(
            f"      {name:24s} pooled OOT PR-AUC={test_results_05[name]['average_precision']:.4f} "
            f"ROC-AUC={test_results_05[name]['roc_auc']:.4f} F1@0.5={test_results_05[name]['f1']:.4f} "
            f"F1@walk-forward-thr={test_results_optimal[name]['f1']:.4f} CVstd={cv_results[name]['average_precision']['std']:.4f}"
        )

    # Persist pooled out-of-time predictions so figures can be rendered without re-training.
    args.deliverables_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.deliverables_dir / "oot_predictions.npz",
        y=y_test_pool.astype(np.int8),
        month=month_test_pool.astype(np.int8),
        **{name: pooled_probs_store[name].astype(np.float32) for name in pooled_probs_store},
    )

    # Final champion: same parsimony rule on ALL out-of-time folds (burn-in + scored)
    ap_all = {n: [average_precision_score(oot_labels[m], oot_probs[n][m]) for m in fold_months] for n in candidates}
    champion_name = select_champion(ap_all)
    champ_threshold = pooled_prod_threshold[champion_name]
    log(f"[4/8] Final champion = {champion_name} | production F1-optimal threshold (pooled OOT) = {champ_threshold:.3f}")

    # ------------------------------------------------------------------------------------------
    # [5/8] Seasonality analysis + slice analysis (champion, scored months)
    # ------------------------------------------------------------------------------------------
    log("[5/8] Seasonality ablation and slice analysis...")
    ablation = {
        "with_calendar": per_month_ap["lightgbm"],
        "without_calendar": per_month_ap[ABLATION_NAME],
        "delta_pr_auc_by_month": {
            k: per_month_ap["lightgbm"][k] - per_month_ap[ABLATION_NAME][k] for k in per_month_ap["lightgbm"]
        },
        "pooled_pr_auc_with": test_results_05["lightgbm"]["average_precision"],
        "pooled_pr_auc_without": test_results_05[ABLATION_NAME]["average_precision"],
    }
    champ_pool = pooled_probs_store[champion_name]
    monthly_calibration = {}
    for m in scored_months:
        mask = month_test_pool == m
        monthly_calibration[str(m)] = {
            "rows": int(mask.sum()),
            "actual_fraud_rate": float(y_test_pool[mask].mean()),
            "mean_predicted_probability": float(champ_pool[mask].mean()),
            "champion_pr_auc": per_month_ap[champion_name][str(m)],
        }
    hist_months = pd.Series(labels).groupby(months_all).mean()
    seasonality = {
        "monthly_fraud_rate_2025": {str(int(k)): float(v) for k, v in hist_months.items()},
        "monthly_calibration_champion": monthly_calibration,
        "calendar_ablation_lightgbm": ablation,
        "regime_checks_by_test_month": {str(k): v for k, v in regime_checks.items()},
        "limitation": (
            "Only one year (2025) is available: December and a generic 'holiday effect' are not separable and "
            "no month other than December shows a distinct seasonal regime. Calendar features can express the "
            "December uplift but cannot be validated out-of-time for any other season."
        ),
    }

    champ_preds_05 = (champ_pool >= 0.5).astype(np.int32)
    slice_analysis = {}
    for slice_col in ["cidade_regiao", "device_type", "tipo_estabelecimento", "turno"]:
        slice_analysis[slice_col] = {}
        for val in sorted(raw_test_pool[slice_col].astype(str).unique()):
            mask = (raw_test_pool[slice_col].astype(str) == val).to_numpy()
            y_s, pred_s = y_test_pool[mask], champ_preds_05[mask]
            slice_analysis[slice_col][str(val)] = {
                "count": int(mask.sum()),
                "fraud_count": int(y_s.sum()),
                "fraud_rate": float(y_s.mean()),
                "precision_at_0_5": float(precision_score(y_s, pred_s, zero_division=0)),
                "recall_at_0_5": float(recall_score(y_s, pred_s, zero_division=0)),
                "f1_at_0_5": float(f1_score(y_s, pred_s, zero_division=0)),
            }

    # ------------------------------------------------------------------------------------------
    # [6/8] Legacy (optimistic) random 80/20 protocol for 3 models, to quantify the gap
    # ------------------------------------------------------------------------------------------
    legacy = {}
    if not args.skip_legacy:
        log("[6/8] Legacy random stratified 80/20 (OPTIMISTIC: random split, threshold tuned on test)...")
        params_legacy = fit_domain_params(dataset)
        x_leg = engineer_features(dataset, params_legacy)
        xd, xt, yd, yt = train_test_split(x_leg, labels, test_size=0.20, random_state=args.seed, stratify=labels)
        leg_models = build_models(args.seed, args.n_estimators, args.n_jobs)
        for name in ["logistic_regression", "xgboost", "lightgbm"]:
            est = leg_models[name]
            est.fit(xd, yd)
            pr = est.predict_proba(xt)[:, 1]
            thr, _ = find_optimal_f1_threshold(yt, pr)
            legacy[name] = {
                "average_precision": float(average_precision_score(yt, pr)),
                "roc_auc": float(roc_auc_score(yt, pr)),
                "f1_at_0_5": float(f1_score(yt, (pr >= 0.5).astype(int), zero_division=0)),
                "f1_at_test_tuned_threshold": float(f1_score(yt, (pr >= thr).astype(int), zero_division=0)),
                "test_tuned_threshold": thr,
            }
            log(f"      legacy {name}: PR-AUC={legacy[name]['average_precision']:.4f} F1@tuned={legacy[name]['f1_at_test_tuned_threshold']:.4f}")
    else:
        log("[6/8] Legacy comparison skipped (--skip-legacy)")

    # ------------------------------------------------------------------------------------------
    # [7/8] Final refit on ALL of 2025 + operational trade-offs + Jan/2026 scoring
    # ------------------------------------------------------------------------------------------
    log("[7/8] Refitting all models on ALL of 2025 (production artifacts)...")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.deliverables_dir.mkdir(parents=True, exist_ok=True)
    final_params = fit_domain_params(dataset)
    (args.models_dir / "domain_params.json").write_text(json.dumps(final_params, indent=2) + "\n", encoding="utf-8")
    x_all = engineer_features(dataset, final_params)
    final_models = build_models(args.seed, args.n_estimators, args.n_jobs)
    fitted_models, operational_tradeoffs = {}, {}
    for name, estimator in final_models.items():
        t0 = time.perf_counter()
        estimator.fit(x_all, labels)
        train_sec = time.perf_counter() - t0
        make_cpu_portable(estimator)
        fitted_models[name] = estimator
        path = args.models_dir / f"{name}_fraud_model_cv.joblib"
        joblib.dump(estimator, path)
        operational_tradeoffs[name] = {
            "train_time_sec": round(train_sec, 2),
            "artifact_size_kb": round(path.stat().st_size / 1024.0, 1),
            **measure_inference_latency_ms(estimator, x_all),
        }
        log(f"      final fit {name:24s} {train_sec:6.1f}s  {operational_tradeoffs[name]['artifact_size_kb']:.0f} KB  p50={operational_tradeoffs[name]['p50_ms']:.2f}ms")
    fresh = build_models(args.seed, args.n_estimators, args.n_jobs)
    voting_clf = VotingClassifier(estimators=[(m, fresh[m]) for m in ENSEMBLE_MEMBERS], voting="soft")
    t0 = time.perf_counter()
    voting_clf.fit(x_all, labels)
    ens_sec = time.perf_counter() - t0
    make_cpu_portable(voting_clf)
    fitted_models[ENSEMBLE_NAME] = voting_clf
    ens_path = args.models_dir / f"{ENSEMBLE_NAME}_fraud_model_cv.joblib"
    joblib.dump(voting_clf, ens_path)
    operational_tradeoffs[ENSEMBLE_NAME] = {
        "train_time_sec": round(ens_sec, 2),
        "artifact_size_kb": round(ens_path.stat().st_size / 1024.0, 1),
        **measure_inference_latency_ms(voting_clf, x_all),
    }
    champion_model = fitted_models[champion_name]
    joblib.dump(champion_model, args.models_dir / "champion_fraud_model_cv.joblib")
    log(f"      champion artifact saved ({champion_name})")

    pre = fitted_models["xgboost"].named_steps["preprocessor"]
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_feat_names = ENGINEERED_NUMERIC_FEATURES + cat_names
    xgb_imp = fitted_models["xgboost"].named_steps["classifier"].feature_importances_
    feat_imp_pairs = sorted(zip(all_feat_names, [float(v) for v in xgb_imp]), key=lambda x: -x[1])

    jan_summary = {}
    if args.jan_data.is_file():
        log(f"[8/8] Scoring January 2026 with champion ({champion_name})...")
        df_jan = pd.read_csv(args.jan_data)
        jan_features = engineer_features(df_jan, final_params)
        jan_probs = champion_model.predict_proba(jan_features)[:, 1]
        df_jan_out = df_jan.copy()
        df_jan_out["flag_fraude"] = (jan_probs >= 0.50).astype(bool)
        df_jan_out["probabilidade_fraude"] = np.round(jan_probs, 6)
        df_jan_out["flag_fraude_threshold_otimo"] = (jan_probs >= champ_threshold).astype(bool)
        if 0.10 < champ_threshold < 0.50:
            edges = [-0.01, 0.10, champ_threshold, 0.50, 1.01]
        else:
            log(f"      WARNING: threshold {champ_threshold:.3f} outside (0.10, 0.50); using fixed tier edges 0.10/0.30/0.50")
            edges = [-0.01, 0.10, 0.30, 0.50, 1.01]
        df_jan_out["nivel_risco"] = pd.cut(jan_probs, bins=edges, labels=["BAIXO", "MODERADO", "ALTO", "CRITICO"]).astype(str)
        jan_out_path = args.deliverables_dir / "transacoes_jan2026_classificadas.csv"
        df_jan_out.to_csv(jan_out_path, index=False)
        log(f"      Saved classified Jan-2026 dataset to: {jan_out_path}")
        jan_summary = {
            "champion_model": champion_name,
            "total_transactions": int(len(df_jan_out)),
            "flagged_at_0_50": int(df_jan_out["flag_fraude"].sum()),
            "flagged_rate_at_0_50": float(df_jan_out["flag_fraude"].mean()),
            "flagged_at_optimal_f1": int(df_jan_out["flag_fraude_threshold_otimo"].sum()),
            "flagged_rate_at_optimal_f1": float(df_jan_out["flag_fraude_threshold_otimo"].mean()),
            "optimal_f1_threshold": float(champ_threshold),
            "expected_fraud_volume_sum_probs": float(jan_probs.sum()),
            "expected_fraud_rate_mean_prob": float(jan_probs.mean()),
            "amount_flagged_at_0_50_brl": float(df_jan_out.loc[df_jan_out["flag_fraude"], "valor_transacao"].sum()),
            "amount_flagged_at_optimal_brl": float(df_jan_out.loc[df_jan_out["flag_fraude_threshold_otimo"], "valor_transacao"].sum()),
            "risk_tier_counts": {k: int(v) for k, v in df_jan_out["nivel_risco"].value_counts().to_dict().items()},
            "sanity": {
                "jan2025_actual_fraud_rate": float(hist_months.get(1, float("nan"))),
                "is_december_mean_must_be_0": float(jan_features["is_december"].mean()),
                "in_regime_share_jan2026": float(jan_features["in_any_fraud_regime"].mean()),
                "in_regime_share_2025": float(x_all["in_any_fraud_regime"].mean()),
                "note": "Jan/2026 has no labels; expected volume assumes the model is calibrated out-of-time (see monthly_calibration_champion).",
            },
        }
        log(f"      Jan-2026: flagged@0.5={jan_summary['flagged_at_0_50']:,} flagged@opt={jan_summary['flagged_at_optimal_f1']:,} expected_rate={jan_summary['expected_fraud_rate_mean_prob']:.3%}")

    log("[8/8] Saving full evaluation metrics report...")
    report = {
        "protocol_version": 2,
        "protocol": {
            "validation": "rolling_origin_walk_forward",
            "burn_in_month": fold_months[0],
            "scored_months": scored_months,
            "threshold_selection": "walk-forward: F1-optimal on pooled out-of-time predictions of PRIOR months only",
            "champion_selection": "walk-forward parsimony rule on prior months' PR-AUC: candidates within 1 fold-std of the best are tied; fixed tie-break order lightgbm > xgboost > catboost > hist_gb > random_forest > logistic_regression > ensemble (hardware independent)",
            "feature_constants": "fit on each fold's training rows only (fit_domain_params)",
            "final_models": "refit on all 2025 rows after selection; threshold = pooled out-of-time F1 optimum",
            "regime_definitions_caveat": "Regime rules originate from full-year EDA (design-time hypothesis); only numeric constants are fit train-only.",
            "bootstrap_note": "Bootstrap CI resamples rows i.i.d. and ignores temporal correlation; treat as optimistic.",
        },
        "champion_model": champion_name,
        "walk_forward_champion_by_month": {str(k): v for k, v in wf_champion.items()},
        "dataset_rows": int(len(dataset)),
        "raw_feature_count": len(RAW_INPUT_COLUMNS),
        "engineered_feature_count": len(ENGINEERED_NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES),
        "encoded_feature_count": len(all_feat_names),
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "engineered_numeric_features": ENGINEERED_NUMERIC_FEATURES,
        "calendar_features": CALENDAR_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "domain_params_final": final_params,
        "class_counts": {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))},
        "split_rows": {
            "scored_test_rows": int(len(y_test_pool)),
            "scored_test_fraud": int(y_test_pool.sum()),
            "folds": fold_info,
        },
        "cross_validation": {
            "strategy": "rolling_origin_walk_forward",
            "fold_count": len(scored_months),
            "fold_months": scored_months,
            "seed": args.seed,
            "threshold": 0.5,
            "metrics": cv_results,
        },
        "test_metrics_at_threshold_0_5": test_results_05,
        "test_metrics_at_optimal_f1_threshold": test_results_optimal,
        "chronological_validation_nov_dec": chrono_results,
        "december_stress_test": december_results,
        "bootstrap_95ci_test": bootstrap_cis,
        "operational_tradeoffs": operational_tradeoffs,
        "calibration_curves": calibration_curves,
        "slice_analysis_champion_test": slice_analysis,
        "seasonality": seasonality,
        "legacy_random_split_optimistic": legacy,
        "top_feature_importances_xgboost": [{"feature": k, "importance": v} for k, v in feat_imp_pairs[:20]],
        "january_2026_forecast": jan_summary,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "execution_environment": {"platform": platform.platform(), "python": platform.python_version(), "gpu_training": USE_GPU},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "joblib": joblib.__version__,
        },
    }
    report_path = args.models_dir / "fraud_models_cv.metrics.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=True) + "\n", encoding="utf-8")
    log(f"DONE. Metrics report saved to: {report_path}")


if __name__ == "__main__":
    main()
