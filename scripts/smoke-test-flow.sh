#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8080}"

python3 - "$API_BASE_URL" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
import uuid

base_url = sys.argv[1].rstrip("/")
suffix = uuid.uuid4().hex[:10]
transaction_id = f"smoke-{suffix}"
payload = {
    "transactionId": transaction_id,
    "cardId": f"Cliente_smoke_{suffix[:4]}",
    "amount": 28.45,
    "timestamp": "2026-01-15 23:15:00",
    "nome_estabelecimento": "Gas Station",
    "tipo_estabelecimento": "Fuel",
    "cidade": "São Paulo",
    "cidade_regiao": "Sudeste",
    "device_type": "mobile",
    "is_new_device": True,
    "temperatura_celsius": 22.0,
    "codigo_postal": "01310-100",
    "score_interno": 510,
    "turno": "noite",
    "valor_categoria": "medio",
}

request = urllib.request.Request(
    f"{base_url}/transactions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        accepted = json.load(response)
except urllib.error.HTTPError as error:
    print(f"Submission failed with HTTP {error.code}: {error.read().decode('utf-8', errors='replace')}")
    raise SystemExit(1)

if accepted.get("transactionId") != transaction_id or accepted.get("status") != "QUEUED":
    raise SystemExit(f"Unexpected submission response: {accepted}")

deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    try:
        with urllib.request.urlopen(f"{base_url}/transactions/{transaction_id}", timeout=5) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"Status query failed with HTTP {error.code}.")
        raise SystemExit(1)

    status = result.get("status")
    print(f"{transaction_id}: {status}")
    if status == "COMPLETED":
        probability = result.get("fraudProbability")
        if not isinstance(probability, (float, int)) or not 0 <= probability <= 1:
            raise SystemExit(f"Invalid fraud probability: {probability}")
        if not isinstance(result.get("isFraud"), bool):
            raise SystemExit("Completed transaction is missing a boolean isFraud result.")
        print(f"End-to-end transaction flow passed (isFraud={result['isFraud']}, prob={probability:.4f}).")
        raise SystemExit(0)
    if status == "FAILED":
        raise SystemExit(f"Transaction entered FAILED: {result.get('message')}")
    time.sleep(0.5)

raise SystemExit(f"Timed out waiting for analysis of {transaction_id}.")
PY
