import json
import os
from math import isfinite
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from train import RAW_INPUT_COLUMNS, engineer_features, load_domain_params

app = FastAPI(title="ATF Fraud Detection Inference API", version="1.0.0")

BASE_DIR = Path(__file__).resolve().parent
REPOSITORY_MODELS_DIR = BASE_DIR.parents[1] / "models"
MODEL_FILES = {
    "champion": REPOSITORY_MODELS_DIR / "champion_fraud_model_cv.joblib",
    "soft_voting_ensemble": REPOSITORY_MODELS_DIR / "soft_voting_ensemble_fraud_model_cv.joblib",
    "catboost": REPOSITORY_MODELS_DIR / "catboost_fraud_model_cv.joblib",
    "xgboost": REPOSITORY_MODELS_DIR / "xgboost_fraud_model_cv.joblib",
    "lightgbm": REPOSITORY_MODELS_DIR / "lightgbm_fraud_model_cv.joblib",
    "hist_gradient_boosting": REPOSITORY_MODELS_DIR / "hist_gradient_boosting_fraud_model_cv.joblib",
    "random_forest": REPOSITORY_MODELS_DIR / "random_forest_fraud_model_cv.joblib",
    "logistic_regression": REPOSITORY_MODELS_DIR / "logistic_regression_fraud_model_cv.joblib",
}
MODEL_NAME = os.getenv("FRAUD_MODEL", "champion").lower()
MODEL_PATH = Path(os.getenv("FRAUD_MODEL_PATH", MODEL_FILES.get(MODEL_NAME, "")))
METRICS_PATH = REPOSITORY_MODELS_DIR / "fraud_models_cv.metrics.json"
DECISION_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
# Constants fit on training data by train.py (falls back to legacy defaults if the file is missing).
DOMAIN_PARAMS = load_domain_params(REPOSITORY_MODELS_DIR / "domain_params.json")

try:
    model = joblib.load(MODEL_PATH) if MODEL_PATH.is_file() else None
except Exception as error:
    print(f"Could not load fraud model at {MODEL_PATH}: {error}")
    model = None


class PredictionRequest(BaseModel):
    transactionId: str | None = None
    cardId: str | None = None
    amount: float | None = None
    timestamp: str | None = None
    datahora_transacao: str | None = None
    nome_estabelecimento: str = "Online Shop 1"
    tipo_estabelecimento: str = "E-commerce"
    valor_transacao: float | None = None
    nome_cliente: str | None = None
    cidade: str = "São Paulo"
    device_type: str = "mobile"
    is_new_device: bool = False
    temperatura_celsius: float = 24.0
    codigo_postal: str = "01000-000"
    score_interno: int = 500
    turno: str = "noite"
    valor_categoria: str = "medio"
    cidade_regiao: str = "Sudeste"
    features: list[float] | None = None


def _build_dataframe_from_request(data: PredictionRequest) -> pd.DataFrame:
    valor = data.valor_transacao if data.valor_transacao is not None else data.amount
    if valor is None:
        if data.features and len(data.features) > 0:
            valor = float(data.features[-1])
        else:
            raise HTTPException(status_code=422, detail="valor_transacao or amount is required.")
    if not isfinite(float(valor)) or float(valor) < 0:
        raise HTTPException(status_code=422, detail="Transaction amount must be a finite non-negative number.")

    if not isfinite(float(data.temperatura_celsius)) or not isfinite(float(data.score_interno)):
        raise HTTPException(status_code=422, detail="Numeric fields must contain finite values.")

    datahora = data.datahora_transacao or data.timestamp or "2026-01-15 22:30:00"
    cliente = data.nome_cliente or data.cardId or "Cliente_0001"

    row = {
        "datahora_transacao": str(datahora),
        "nome_estabelecimento": str(data.nome_estabelecimento),
        "tipo_estabelecimento": str(data.tipo_estabelecimento),
        "valor_transacao": float(valor),
        "nome_cliente": str(cliente),
        "cidade": str(data.cidade),
        "device_type": str(data.device_type),
        "is_new_device": bool(data.is_new_device),
        "temperatura_celsius": float(data.temperatura_celsius),
        "codigo_postal": str(data.codigo_postal),
        "score_interno": int(data.score_interno),
        "turno": str(data.turno),
        "valor_categoria": str(data.valor_categoria),
        "cidade_regiao": str(data.cidade_regiao),
    }
    return engineer_features(pd.DataFrame([row])[RAW_INPUT_COLUMNS], DOMAIN_PARAMS)


@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Fraud model is not loaded.")
    return {
        "status": "UP",
        "model_loaded": True,
        "model": MODEL_NAME,
        "threshold": DECISION_THRESHOLD,
    }


@app.get("/metrics")
def model_metrics():
    if model is None:
        raise HTTPException(status_code=503, detail="Fraud model is not loaded.")
    if not METRICS_PATH.is_file():
        raise HTTPException(status_code=404, detail="Model evaluation report is not available.")
    try:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Could not read model evaluation report.") from error
    return {**report, "active_model": MODEL_NAME}


@app.post("/predict")
def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Fraud model is not loaded.")

    if data.features is not None:
        if not data.features or not all(isfinite(value) for value in data.features):
            raise HTTPException(status_code=422, detail="features must contain finite numeric values.")
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features is not None and len(data.features) != expected_features:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {expected_features} features, received {len(data.features)}.",
            )
        feature_input = np.asarray(data.features, dtype=float).reshape(1, -1)
    else:
        feature_input = _build_dataframe_from_request(data)

    try:
        probabilities = model.predict_proba(feature_input)[0]
        classes = list(model.classes_)
        fraud_class_index = classes.index(1)
        fraud_prob = float(probabilities[fraud_class_index])
        is_fraud = bool(fraud_prob >= DECISION_THRESHOLD)
    except (AttributeError, ValueError, IndexError, TypeError) as error:
        raise HTTPException(status_code=500, detail="Loaded model has an incompatible prediction interface.") from error

    return {
        "isFraud": is_fraud,
        "fraudProbability": fraud_prob,
    }
