#!/usr/bin/env python3
"""
Stream a sample of January-2026 transactions from deliverables/transacoes_jan2026_classificadas.csv
(or data/transacoes_jan2026.csv) into the Transaction Gateway -> Kafka -> Fraud Analyzer -> FastAPI ML pipeline.
"""
import csv
import json
import random
import sys
import uuid
from pathlib import Path
from urllib.request import Request, urlopen

GATEWAY_URL = "http://localhost:8080/transactions"
BASE_DIR = Path(__file__).resolve().parents[1]
SCORED_CSV_PATH = BASE_DIR / "deliverables" / "transacoes_jan2026_classificadas.csv"
RAW_CSV_PATH = BASE_DIR / "data" / "transacoes_jan2026.csv"


def load_rows():
    csv_path = SCORED_CSV_PATH if SCORED_CSV_PATH.is_file() else RAW_CSV_PATH
    if not csv_path.is_file():
        raise SystemExit(f"Dataset not found at {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_payload_from_row(transaction_id: str, row: dict):
    amount = float(row["valor_transacao"])
    is_new_dev = str(row["is_new_device"]).strip().lower() == "true"
    return {
        "transactionId": transaction_id,
        "cardId": row["nome_cliente"],
        "amount": amount,
        "timestamp": row["datahora_transacao"],
        "nome_estabelecimento": row["nome_estabelecimento"],
        "tipo_estabelecimento": row["tipo_estabelecimento"],
        "cidade": row["cidade"],
        "cidade_regiao": row["cidade_regiao"],
        "device_type": row["device_type"],
        "is_new_device": is_new_dev,
        "temperatura_celsius": float(row["temperatura_celsius"]),
        "codigo_postal": row["codigo_postal"],
        "score_interno": int(float(row["score_interno"])),
        "turno": row["turno"],
        "valor_categoria": row["valor_categoria"],
    }


def send_transaction(payload: dict):
    body = json.dumps(payload).encode("utf-8")
    req = Request(GATEWAY_URL, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=20) as resp:
            response_text = resp.read().decode("utf-8")
            print(f"{payload['transactionId']} ({payload['nome_estabelecimento']} / {payload['cidade']}) -> {resp.status} {response_text}")
    except Exception as exc:
        print(f"{payload['transactionId']} -> ERROR: {exc}")


def main():
    requested_count = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    rows = load_rows()
    if not rows:
        raise SystemExit("No rows found in CSV.")

    fraud_rows = [r for r in rows if str(r.get("flag_fraude", "")).lower() == "true"]
    legit_rows = [r for r in rows if str(r.get("flag_fraude", "")).lower() != "true"]

    if fraud_rows and legit_rows:
        n_fraud = min(len(fraud_rows), max(1, requested_count // 3))
        n_legit = min(len(legit_rows), max(1, requested_count - n_fraud))
        selected = random.sample(fraud_rows, n_fraud) + random.sample(legit_rows, n_legit)
        random.shuffle(selected)
    else:
        selected = random.sample(rows, min(len(rows), requested_count))

    for row in selected[:requested_count]:
        suffix = uuid.uuid4().hex[:8]
        tag = "risk" if str(row.get("flag_fraude", "")).lower() == "true" else "legit"
        transaction_id = f"jan26-{tag}-{suffix}"
        payload = build_payload_from_row(transaction_id, row)
        send_transaction(payload)


if __name__ == "__main__":
    main()
