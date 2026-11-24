import numpy as np
from fastapi.testclient import TestClient

import main


class FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, features):
        return np.array([1])

    def predict_proba(self, features):
        return np.array([[0.12, 0.88]])


class FakeVectorModel(FakeModel):
    n_features_in_ = 3


client = TestClient(main.app)


def test_health_reports_loaded_model(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_accepts_case_domain_payload(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    payload = {
        "transactionId": "txn-case-001",
        "cardId": "Cliente_16241",
        "amount": 28.50,
        "timestamp": "2026-01-15 22:30:00",
        "nome_estabelecimento": "Gas Station",
        "tipo_estabelecimento": "Fuel",
        "cidade": "São Paulo",
        "device_type": "mobile",
        "is_new_device": True,
        "temperatura_celsius": 22.5,
        "codigo_postal": "01310-100",
        "score_interno": 512,
        "turno": "noite",
        "valor_categoria": "medio",
        "cidade_regiao": "Sudeste",
    }
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"isFraud": True, "fraudProbability": 0.88}


def test_predict_returns_fraud_class_and_probability_for_vector(monkeypatch):
    monkeypatch.setattr(main, "model", FakeVectorModel())

    response = client.post("/predict", json={"features": [0.1, 0.2, 0.3]})

    assert response.status_code == 200
    assert response.json() == {"isFraud": True, "fraudProbability": 0.88}


def test_predict_rejects_wrong_feature_count(monkeypatch):
    monkeypatch.setattr(main, "model", FakeVectorModel())

    response = client.post("/predict", json={"features": [0.1, 0.2]})

    assert response.status_code == 422


def test_predict_reports_unavailable_model(monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.post("/predict", json={"amount": 25.0})

    assert response.status_code == 503


def test_metrics_returns_evaluation_report(monkeypatch, tmp_path):
    report_path = tmp_path / "metrics.json"
    report_path.write_text('{"cross_validation": {"fold_count": 5}}')
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "METRICS_PATH", report_path)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["active_model"] == main.MODEL_NAME
    assert response.json()["cross_validation"]["fold_count"] == 5


def test_metrics_reports_missing_model_report(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "METRICS_PATH", tmp_path / "missing.json")

    response = client.get("/metrics")

    assert response.status_code == 404


# ---- Seasonality / feature-engineering tests (no model needed) ----
import pandas as pd

from train import CALENDAR_FEATURES, ENGINEERED_NUMERIC_FEATURES, engineer_features, fit_domain_params


def _row(ts, tipo="E-commerce", hour_ts=None):
    return {
        "datahora_transacao": ts,
        "nome_estabelecimento": "Online Shop 1",
        "tipo_estabelecimento": tipo,
        "valor_transacao": 30.0,
        "nome_cliente": "Cliente_1",
        "cidade": "São Paulo",
        "device_type": "mobile",
        "is_new_device": True,
        "temperatura_celsius": 24.0,
        "codigo_postal": "01000-000",
        "score_interno": 500,
        "turno": "noite",
        "valor_categoria": "medio",
        "cidade_regiao": "Sudeste",
    }


def test_calendar_features_present_and_december_flags():
    dec = engineer_features(pd.DataFrame([_row("2025-12-20 21:00:00")]))
    jan = engineer_features(pd.DataFrame([_row("2026-01-20 21:00:00")]))
    assert set(CALENDAR_FEATURES) <= set(dec.columns)
    assert list(dec.columns[: len(ENGINEERED_NUMERIC_FEATURES)]) == ENGINEERED_NUMERIC_FEATURES
    assert dec.loc[0, "is_december"] == 1 and jan.loc[0, "is_december"] == 0
    assert dec.loc[0, "days_to_christmas"] == 5
    assert jan.loc[0, "days_to_christmas"] == 31  # January must never look like Christmas week
    assert dec.loc[0, "regime_3_ecom_dez_noite"] == 1 and jan.loc[0, "regime_3_ecom_dez_noite"] == 0


def test_holiday_and_eve_flags():
    out = engineer_features(pd.DataFrame([_row("2025-12-25 10:00:00"), _row("2025-12-24 10:00:00")]))
    assert out.loc[0, "is_holiday"] == 1 and out.loc[1, "is_holiday_eve"] == 1


def test_fit_domain_params_uses_only_given_rows():
    df = pd.DataFrame({"valor_transacao": [10.0] * 60 + [100.0] * 60, "flag_fraude": [True] * 60 + [False] * 60})
    params = fit_domain_params(df)
    assert params["fraud_value_median"] == 10.0  # legit rows (100.0) must not influence fraud constants
