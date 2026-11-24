# Local Demo Run Guide (Fraud Prevention Case — January/2026)

This guide explains how to start the full architecture (React Dashboard + Spring Boot Gateway + Apache Kafka + Spring Boot Fraud Analyzer + FastAPI ML Service + PostgreSQL) for the case presentation.

## 1. Prerequisites and Artifact Check (`.joblib`)

> The `.joblib` models are not published in the repository. Generate them first with `GPU=none bash scripts/colab_train_and_score.sh` (section 2).

From the root of the `atf-fraud-detection-case` repository:

```bash
test -f models/champion_fraud_model_cv.joblib
test -f models/lightgbm_fraud_model_cv.joblib
test -f models/xgboost_fraud_model_cv.joblib
test -f models/logistic_regression_fraud_model_cv.joblib
test -f models/domain_params.json
test -f models/fraud_models_cv.metrics.json
test -f Case_Fraude_Desenvolvimento.ipynb
```

## 2. Retrain on Colab (optional) and Follow the Log

```bash
GPU=T4 bash scripts/colab_train_and_score.sh      # GPU=none for a CPU-only VM
```

Heavy jobs always run on Colab (`JOB=ablation bash scripts/colab_train_and_score.sh` runs the protocol ablation on a CPU VM). Training runs as a background process on the VM and writes `/content/train.log`; the script streams it to your terminal. To follow it from a second terminal:

```bash
watch -n 10 'colab exec -s atf-fraud -f scripts/colab_tail.py --env N=30'
colab log -s atf-fraud -n 20      # session event history
colab status -s atf-fraud
```

## 3. Configure `.env` and Start the Containers

```bash
cp .env.example .env
# Edit .env and set a strong local password in POSTGRES_PASSWORD
docker compose up -d --build
docker compose ps
```

All exposed services bind to `127.0.0.1` only:
- **Dashboard:** `http://localhost:3000`
- **Transaction Gateway:** `http://localhost:8080`
- **FastAPI ML Service:** `http://localhost:8000/docs`

## 4. Simulate the Real-Time Flow (January/2026)

1. Open `http://localhost:3000` and click **New demo transaction** to submit a custom transaction (choose merchant, city, device type, new-device flag and time).
2. Or fire a batch of real January/2026 transactions (`deliverables/transacoes_jan2026_classificadas.csv`) from the terminal:

```bash
python3 scripts/send_demo_transactions.py 30
```

3. Validate the end-to-end flow with the smoke test:

```bash
bash scripts/smoke-test-flow.sh
```

## 5. Troubleshooting: NordVPN (or another VPN) and Docker

Symptom: `docker compose up` succeeds and the ports show as listening, but `curl http://localhost:8000/health` hangs, and the `transaction-gateway` log stays on `Waiting for PostgreSQL at postgres:5432`.

Cause: with the VPN's firewall on (Kill Switch on, **LAN Discovery off**) the VPN drops traffic to private ranges, which includes Docker's bridge network (`172.16.0.0/12`). Even container-to-container traffic is blocked, so Kafka, Postgres and the services cannot reach each other.

Fix (narrow and reversible; keeps the VPN and Kill Switch on):

```bash
nordvpn allowlist add subnet 172.16.0.0/12
# undo later with:
nordvpn allowlist remove subnet 172.16.0.0/12
```

Verify with `bash scripts/smoke-test-flow.sh` (expects `QUEUED -> PROCESSING -> COMPLETED`). Turning on `nordvpn set lan-discovery on` also works but opens all private ranges, so the allowlist is preferred.

## 6. Stop the Stack

```bash
docker compose down
```
