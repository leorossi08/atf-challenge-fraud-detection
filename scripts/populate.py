#!/usr/bin/env python3
"""
Streams a rich, diverse set of synthetic Brazilian merchant transactions into the
Transaction Gateway -> Apache Kafka -> Fraud Analyzer -> FastAPI ML pipeline.

Supports:
- 45+ curated scenarios spanning all 4 fraud regimes (R1..R4), Step-Up (2FA) boundary cases,
  high-value safe purchases, edge-case tickets, and all cities/categories/devices
  (desktop, mobile, tablet; known vs new device).
- Burst / concurrent streaming (--concurrency / --burst) so you can visibly watch
  QUEUED -> PROCESSING -> COMPLETED transitions in real time on http://localhost:3000.
- Continuous live stream mode (--stream) that keeps feeding realistic transactions
  until Ctrl+C.

Usage:
    python3 scripts/populate.py                         # 50 transactions in bursts (creates visible queue)
    python3 scripts/populate.py --count 80 --burst 10   # 80 transactions in bursts of 10
    python3 scripts/populate.py --stream --rate 1.5     # continuous live stream (~1.5 tx/sec)
"""
import argparse
import concurrent.futures
import json
import random
import time
import uuid
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

GATEWAY_URL = "http://127.0.0.1:8080/transactions"

CURATED_SCENARIOS = [
    # --- REGIME 1: Fuel 00h-08h in São Paulo / Rio de Janeiro ---
    {"tag": "r1-fuel-sp-newmob", "cardId": "Cliente_90101", "amount": 28.90, "timestamp": "2026-01-12 03:15:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 21.4, "codigo_postal": "01310-100", "score_interno": 612, "turno": "madrugada", "valor_categoria": "medio"},
    {"tag": "r1-fuel-rj-newmob", "cardId": "Cliente_90102", "amount": 34.50, "timestamp": "2026-01-14 06:40:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 27.8, "codigo_postal": "22041-001", "score_interno": 488, "turno": "manha", "valor_categoria": "medio"},
    {"tag": "r1-fuel-sp-newdesk", "cardId": "Cliente_90103", "amount": 22.10, "timestamp": "2026-01-18 02:05:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "desktop", "is_new_device": True, "temperatura_celsius": 19.9, "codigo_postal": "04543-011", "score_interno": 730, "turno": "madrugada", "valor_categoria": "baixo"},
    {"tag": "r1-fuel-rj-newtab", "cardId": "Cliente_90105", "amount": 29.80, "timestamp": "2026-01-20 01:45:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "tablet", "is_new_device": True, "temperatura_celsius": 26.5, "codigo_postal": "22640-102", "score_interno": 310, "turno": "madrugada", "valor_categoria": "medio"},
    {"tag": "r1-fuel-sp-stepup", "cardId": "Cliente_90106", "amount": 31.40, "timestamp": "2026-01-21 05:10:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 20.1, "codigo_postal": "05424-000", "score_interno": 590, "turno": "madrugada", "valor_categoria": "medio"},
    {"tag": "r1-fuel-bh-safe", "cardId": "Cliente_90107", "amount": 33.00, "timestamp": "2026-01-22 04:15:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "Belo Horizonte", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 23.0, "codigo_postal": "30130-110", "score_interno": 450, "turno": "madrugada", "valor_categoria": "medio"},
    {"tag": "r1-fuel-recife-safe", "cardId": "Cliente_90104", "amount": 31.20, "timestamp": "2026-01-18 04:20:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "Recife", "cidade_regiao": "Nordeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 29.5, "codigo_postal": "50010-000", "score_interno": 420, "turno": "madrugada", "valor_categoria": "medio"},
    {"tag": "r1-fuel-sp-highval", "cardId": "Cliente_90108", "amount": 285.00, "timestamp": "2026-01-23 07:30:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 22.4, "codigo_postal": "01451-000", "score_interno": 880, "turno": "manha", "valor_categoria": "alto"},

    # --- REGIME 2: Food & Beverage 12h-14h in Sul / Sudeste ---
    {"tag": "r2-food-bh-newmob", "cardId": "Cliente_90201", "amount": 27.40, "timestamp": "2026-01-15 12:35:00", "nome_estabelecimento": "Restaurant X", "tipo_estabelecimento": "Food & Beverage", "cidade": "Belo Horizonte", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 24.6, "codigo_postal": "30130-110", "score_interno": 510, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "r2-food-cwb-newmob", "cardId": "Cliente_90202", "amount": 36.80, "timestamp": "2026-01-16 13:10:00", "nome_estabelecimento": "Restaurant Y", "tipo_estabelecimento": "Food & Beverage", "cidade": "Curitiba", "cidade_regiao": "Sul", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 18.2, "codigo_postal": "80010-000", "score_interno": 815, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "r2-food-sp-newdesk", "cardId": "Cliente_90205", "amount": 24.90, "timestamp": "2026-01-17 12:50:00", "nome_estabelecimento": "Restaurant X", "tipo_estabelecimento": "Food & Beverage", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "desktop", "is_new_device": True, "temperatura_celsius": 23.8, "codigo_postal": "01311-200", "score_interno": 210, "turno": "tarde", "valor_categoria": "baixo"},
    {"tag": "r2-food-rj-newtab", "cardId": "Cliente_90206", "amount": 33.20, "timestamp": "2026-01-18 14:20:00", "nome_estabelecimento": "Restaurant Y", "tipo_estabelecimento": "Food & Beverage", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "tablet", "is_new_device": True, "temperatura_celsius": 31.2, "codigo_postal": "22250-040", "score_interno": 605, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "r2-food-poa-stepup", "cardId": "Cliente_90203", "amount": 29.50, "timestamp": "2026-01-19 14:05:00", "nome_estabelecimento": "Restaurant X", "tipo_estabelecimento": "Food & Beverage", "cidade": "Porto Alegre", "cidade_regiao": "Sul", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 22.0, "codigo_postal": "90010-150", "score_interno": 390, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "r2-food-cwb-knowndesk", "cardId": "Cliente_90207", "amount": 21.50, "timestamp": "2026-01-20 13:40:00", "nome_estabelecimento": "Restaurant Y", "tipo_estabelecimento": "Food & Beverage", "cidade": "Curitiba", "cidade_regiao": "Sul", "device_type": "desktop", "is_new_device": False, "temperatura_celsius": 17.5, "codigo_postal": "80420-090", "score_interno": 495, "turno": "tarde", "valor_categoria": "baixo"},
    {"tag": "r2-food-ssa-safe", "cardId": "Cliente_90204", "amount": 32.10, "timestamp": "2026-01-19 13:25:00", "nome_estabelecimento": "Restaurant Y", "tipo_estabelecimento": "Food & Beverage", "cidade": "Salvador", "cidade_regiao": "Nordeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 30.1, "codigo_postal": "40020-000", "score_interno": 505, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "r2-food-rec-safe", "cardId": "Cliente_90208", "amount": 28.00, "timestamp": "2026-01-21 12:15:00", "nome_estabelecimento": "Restaurant X", "tipo_estabelecimento": "Food & Beverage", "cidade": "Recife", "cidade_regiao": "Nordeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 30.4, "codigo_postal": "51020-000", "score_interno": 740, "turno": "tarde", "valor_categoria": "medio"},

    # --- REGIME 4: Night 22h-23h Across All Sectors & Cities ---
    {"tag": "r4-ecom-sp-newmob", "cardId": "Cliente_90401", "amount": 25.90, "timestamp": "2026-01-20 22:18:00", "nome_estabelecimento": "Online Shop 1", "tipo_estabelecimento": "E-commerce", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 21.8, "codigo_postal": "01415-001", "score_interno": 640, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-ecom-poa-newmob", "cardId": "Cliente_90407", "amount": 33.40, "timestamp": "2026-01-21 23:12:00", "nome_estabelecimento": "Online Shop 2", "tipo_estabelecimento": "E-commerce", "cidade": "Porto Alegre", "cidade_regiao": "Sul", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 19.0, "codigo_postal": "90450-190", "score_interno": 150, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-retail-rj-newmob", "cardId": "Cliente_90402", "amount": 38.75, "timestamp": "2026-01-21 23:04:00", "nome_estabelecimento": "Electronics Store", "tipo_estabelecimento": "Retail", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 28.4, "codigo_postal": "20040-002", "score_interno": 270, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-cloth-bh-newmob", "cardId": "Cliente_90408", "amount": 27.15, "timestamp": "2026-01-22 22:33:00", "nome_estabelecimento": "Clothing Store", "tipo_estabelecimento": "Retail", "cidade": "Belo Horizonte", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 23.2, "codigo_postal": "30140-071", "score_interno": 820, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-pharma-rec-newdesk", "cardId": "Cliente_90403", "amount": 24.30, "timestamp": "2026-01-22 22:47:00", "nome_estabelecimento": "Pharmacy Z", "tipo_estabelecimento": "Pharmacy", "cidade": "Recife", "cidade_regiao": "Nordeste", "device_type": "desktop", "is_new_device": True, "temperatura_celsius": 28.9, "codigo_postal": "51020-000", "score_interno": 530, "turno": "noite", "valor_categoria": "baixo"},
    {"tag": "r4-super-cwb-newtab", "cardId": "Cliente_90404", "amount": 31.60, "timestamp": "2026-01-23 23:22:00", "nome_estabelecimento": "Supermarket A", "tipo_estabelecimento": "Supermarket", "cidade": "Curitiba", "cidade_regiao": "Sul", "device_type": "tablet", "is_new_device": True, "temperatura_celsius": 16.5, "codigo_postal": "80420-090", "score_interno": 702, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-super-sp-newmob", "cardId": "Cliente_90409", "amount": 29.00, "timestamp": "2026-01-24 22:09:00", "nome_estabelecimento": "Supermarket B", "tipo_estabelecimento": "Supermarket", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 21.0, "codigo_postal": "04094-050", "score_interno": 440, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-fuel-ssa-newmob", "cardId": "Cliente_90410", "amount": 35.20, "timestamp": "2026-01-25 23:41:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "Salvador", "cidade_regiao": "Nordeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 28.1, "codigo_postal": "41820-020", "score_interno": 330, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-food-ssa-newmob", "cardId": "Cliente_90510", "amount": 29.90, "timestamp": "2026-01-30 23:45:00", "nome_estabelecimento": "Restaurant Y", "tipo_estabelecimento": "Food & Beverage", "cidade": "Salvador", "cidade_regiao": "Nordeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 28.2, "codigo_postal": "40140-130", "score_interno": 355, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-cloth-poa-knownmob", "cardId": "Cliente_90405", "amount": 26.80, "timestamp": "2026-01-24 22:55:00", "nome_estabelecimento": "Clothing Store", "tipo_estabelecimento": "Retail", "cidade": "Porto Alegre", "cidade_regiao": "Sul", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 19.3, "codigo_postal": "90560-002", "score_interno": 450, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "r4-pharma-sp-knowntab", "cardId": "Cliente_90411", "amount": 18.50, "timestamp": "2026-01-26 22:15:00", "nome_estabelecimento": "Pharmacy Z", "tipo_estabelecimento": "Pharmacy", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "tablet", "is_new_device": False, "temperatura_celsius": 22.0, "codigo_postal": "01310-100", "score_interno": 690, "turno": "noite", "valor_categoria": "baixo"},
    {"tag": "r4-highval-safe", "cardId": "Cliente_90406", "amount": 420.00, "timestamp": "2026-01-24 23:10:00", "nome_estabelecimento": "Electronics Store", "tipo_estabelecimento": "Retail", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "desktop", "is_new_device": False, "temperatura_celsius": 22.1, "codigo_postal": "04094-050", "score_interno": 890, "turno": "noite", "valor_categoria": "alto"},
    {"tag": "r4-microticket-safe", "cardId": "Cliente_90412", "amount": 3.50, "timestamp": "2026-01-27 22:30:00", "nome_estabelecimento": "Supermarket A", "tipo_estabelecimento": "Supermarket", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 27.5, "codigo_postal": "22041-001", "score_interno": 500, "turno": "noite", "valor_categoria": "baixo"},

    # --- SAFE / LEGITIMATE TRANSACTIONS (Diverse Channels, Cities, Tickets & Hours) ---
    {"tag": "safe-morning-super-desk", "cardId": "Cliente_90501", "amount": 145.80, "timestamp": "2026-01-10 09:30:00", "nome_estabelecimento": "Supermarket B", "tipo_estabelecimento": "Supermarket", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "desktop", "is_new_device": False, "temperatura_celsius": 23.0, "codigo_postal": "05424-000", "score_interno": 775, "turno": "manha", "valor_categoria": "alto"},
    {"tag": "safe-morning-pharma-mob", "cardId": "Cliente_90502", "amount": 19.90, "timestamp": "2026-01-11 10:15:00", "nome_estabelecimento": "Pharmacy Z", "tipo_estabelecimento": "Pharmacy", "cidade": "Belo Horizonte", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 25.1, "codigo_postal": "30140-071", "score_interno": 520, "turno": "manha", "valor_categoria": "baixo"},
    {"tag": "safe-morning-elec-desk", "cardId": "Cliente_90511", "amount": 589.90, "timestamp": "2026-01-12 10:50:00", "nome_estabelecimento": "Electronics Store", "tipo_estabelecimento": "Retail", "cidade": "Curitiba", "cidade_regiao": "Sul", "device_type": "desktop", "is_new_device": False, "temperatura_celsius": 19.0, "codigo_postal": "80010-000", "score_interno": 910, "turno": "manha", "valor_categoria": "alto"},
    {"tag": "safe-afternoon-ecom-tab", "cardId": "Cliente_90503", "amount": 89.90, "timestamp": "2026-01-13 15:45:00", "nome_estabelecimento": "Online Shop 2", "tipo_estabelecimento": "E-commerce", "cidade": "Curitiba", "cidade_regiao": "Sul", "device_type": "tablet", "is_new_device": False, "temperatura_celsius": 19.7, "codigo_postal": "80250-210", "score_interno": 640, "turno": "tarde", "valor_categoria": "alto"},
    {"tag": "safe-afternoon-cloth-desk", "cardId": "Cliente_90504", "amount": 54.00, "timestamp": "2026-01-17 16:20:00", "nome_estabelecimento": "Clothing Store", "tipo_estabelecimento": "Retail", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "desktop", "is_new_device": False, "temperatura_celsius": 29.2, "codigo_postal": "22250-040", "score_interno": 310, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "safe-afternoon-fuel-tab", "cardId": "Cliente_90507", "amount": 74.20, "timestamp": "2026-01-27 17:10:00", "nome_estabelecimento": "Gas Station", "tipo_estabelecimento": "Fuel", "cidade": "Recife", "cidade_regiao": "Nordeste", "device_type": "tablet", "is_new_device": False, "temperatura_celsius": 30.8, "codigo_postal": "52011-000", "score_interno": 710, "turno": "tarde", "valor_categoria": "medio"},
    {"tag": "safe-afternoon-super-mob", "cardId": "Cliente_90512", "amount": 112.40, "timestamp": "2026-01-22 15:10:00", "nome_estabelecimento": "Supermarket A", "tipo_estabelecimento": "Supermarket", "cidade": "Porto Alegre", "cidade_regiao": "Sul", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 21.5, "codigo_postal": "90010-150", "score_interno": 665, "turno": "tarde", "valor_categoria": "alto"},
    {"tag": "safe-evening-food-mob", "cardId": "Cliente_90505", "amount": 62.50, "timestamp": "2026-01-25 19:30:00", "nome_estabelecimento": "Restaurant X", "tipo_estabelecimento": "Food & Beverage", "cidade": "Salvador", "cidade_regiao": "Nordeste", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 28.5, "codigo_postal": "41820-020", "score_interno": 580, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "safe-evening-ecom-desk", "cardId": "Cliente_90506", "amount": 215.00, "timestamp": "2026-01-26 20:15:00", "nome_estabelecimento": "Online Shop 1", "tipo_estabelecimento": "E-commerce", "cidade": "Porto Alegre", "cidade_regiao": "Sul", "device_type": "desktop", "is_new_device": True, "temperatura_celsius": 20.4, "codigo_postal": "90450-190", "score_interno": 495, "turno": "noite", "valor_categoria": "alto"},
    {"tag": "safe-midday-super-mob", "cardId": "Cliente_90508", "amount": 18.40, "timestamp": "2026-01-28 11:05:00", "nome_estabelecimento": "Supermarket A", "tipo_estabelecimento": "Supermarket", "cidade": "Belo Horizonte", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": False, "temperatura_celsius": 24.0, "codigo_postal": "30112-000", "score_interno": 625, "turno": "manha", "valor_categoria": "baixo"},
    {"tag": "safe-evening-pharma-tab", "cardId": "Cliente_90509", "amount": 44.90, "timestamp": "2026-01-29 18:40:00", "nome_estabelecimento": "Pharmacy Z", "tipo_estabelecimento": "Pharmacy", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "tablet", "is_new_device": False, "temperatura_celsius": 22.7, "codigo_postal": "01311-200", "score_interno": 410, "turno": "noite", "valor_categoria": "medio"},
    {"tag": "safe-evening-cloth-tab", "cardId": "Cliente_90513", "amount": 129.00, "timestamp": "2026-01-30 21:10:00", "nome_estabelecimento": "Clothing Store", "tipo_estabelecimento": "Retail", "cidade": "Recife", "cidade_regiao": "Nordeste", "device_type": "tablet", "is_new_device": False, "temperatura_celsius": 28.8, "codigo_postal": "50010-000", "score_interno": 530, "turno": "noite", "valor_categoria": "alto"},
    {"tag": "safe-morning-ecom-mob", "cardId": "Cliente_90514", "amount": 42.30, "timestamp": "2026-01-31 09:05:00", "nome_estabelecimento": "Online Shop 2", "tipo_estabelecimento": "E-commerce", "cidade": "Rio de Janeiro", "cidade_regiao": "Sudeste", "device_type": "mobile", "is_new_device": True, "temperatura_celsius": 27.1, "codigo_postal": "20040-002", "score_interno": 190, "turno": "manha", "valor_categoria": "medio"},
    {"tag": "safe-afternoon-food-desk", "cardId": "Cliente_90515", "amount": 38.00, "timestamp": "2026-01-31 16:45:00", "nome_estabelecimento": "Restaurant Y", "tipo_estabelecimento": "Food & Beverage", "cidade": "São Paulo", "cidade_regiao": "Sudeste", "device_type": "desktop", "is_new_device": False, "temperatura_celsius": 24.4, "codigo_postal": "04543-011", "score_interno": 760, "turno": "tarde", "valor_categoria": "medio"},
]

ESTABLISHMENTS = [
    ("Gas Station", "Fuel"),
    ("Restaurant X", "Food & Beverage"),
    ("Restaurant Y", "Food & Beverage"),
    ("Online Shop 1", "E-commerce"),
    ("Online Shop 2", "E-commerce"),
    ("Electronics Store", "Retail"),
    ("Clothing Store", "Retail"),
    ("Supermarket A", "Supermarket"),
    ("Supermarket B", "Supermarket"),
    ("Pharmacy Z", "Pharmacy"),
]

CITIES = [
    ("São Paulo", "Sudeste", 22.0),
    ("Rio de Janeiro", "Sudeste", 28.0),
    ("Belo Horizonte", "Sudeste", 24.0),
    ("Curitiba", "Sul", 18.0),
    ("Porto Alegre", "Sul", 17.0),
    ("Recife", "Nordeste", 30.0),
    ("Salvador", "Nordeste", 29.0),
]


def generate_random_scenario(idx: int) -> dict:
    name, kind = random.choice(ESTABLISHMENTS)
    city, region, base_temp = random.choice(CITIES)
    device = random.choices(["mobile", "desktop", "tablet"], weights=[0.50, 0.32, 0.18])[0]
    is_new = random.random() < 0.35
    hour = random.choices([2, 6, 10, 13, 16, 19, 22, 23], weights=[1, 1, 2, 2, 2, 2, 3, 3])[0]
    day = random.randint(1, 31)
    minute = random.randint(0, 59)
    amount = round(random.uniform(14.0, 46.0) if random.random() < 0.7 else random.uniform(60.0, 380.0), 2)
    turno = "madrugada" if hour < 6 else "manha" if hour < 12 else "tarde" if hour < 18 else "noite"
    val_cat = "baixo" if amount < 25.0 else "medio" if amount <= 80.0 else "alto"
    return {
        "tag": f"stream-{idx:03d}",
        "cardId": f"Cliente_{random.randint(10000, 99999)}",
        "amount": amount,
        "timestamp": f"2026-01-{day:02d} {hour:02d}:{minute:02d}:00",
        "nome_estabelecimento": name,
        "tipo_estabelecimento": kind,
        "cidade": city,
        "cidade_regiao": region,
        "device_type": device,
        "is_new_device": is_new,
        "temperatura_celsius": round(base_temp + random.uniform(-3.0, 3.0), 1),
        "codigo_postal": f"{random.randint(10000, 99999)}-{random.randint(100, 999)}",
        "score_interno": random.randint(0, 1000),
        "turno": turno,
        "valor_categoria": val_cat,
    }


def send_transaction(gateway_url: str, scenario: dict) -> bool:
    suffix = uuid.uuid4().hex[:6]
    tx_id = f"pop-{scenario['tag']}-{suffix}"
    payload = {k: v for k, v in scenario.items() if k != "tag"}
    payload["transactionId"] = tx_id

    req = Request(
        gateway_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            dev_badge = f"{payload['device_type']}{'+NEW' if payload['is_new_device'] else ''}"
            print(
                f"[QUEUED {resp.status}] {tx_id:30s} | {payload['nome_estabelecimento']:17s} | "
                f"{payload['cidade']:14s} | {dev_badge:11s} | R$ {payload['amount']:6.2f} -> {body}",
                flush=True,
            )
            return True
    except (HTTPError, URLError) as err:
        print(f"[ERR] {tx_id} -> {err}", flush=True)
        return False


def send_burst(gateway_url: str, items: list[dict], workers: int) -> int:
    ok = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(send_transaction, gateway_url, sc) for sc in items]
        for fut in concurrent.futures.as_completed(futures):
            if fut.result():
                ok += 1
    return ok


def main():
    parser = argparse.ArgumentParser(description="Stream synthetic transactions into the Kafka fraud pipeline.")
    parser.add_argument("--url", default=GATEWAY_URL, help="Transaction Gateway POST /transactions URL")
    parser.add_argument("--count", type=int, default=60, help="Total transactions in batch mode (default: 60)")
    parser.add_argument("--burst", type=int, default=6, help="Transactions sent concurrently per wave to build a visible queue (default: 6)")
    parser.add_argument("--wave-pause", type=float, default=1.2, help="Pause in seconds between waves (default: 1.2s)")
    parser.add_argument("--stream", action="store_true", help="Continuous live stream mode (Ctrl+C to stop)")
    parser.add_argument("--rate", type=float, default=1.5, help="Average transactions/sec in --stream mode")
    args = parser.parse_args()

    if args.stream:
        print(f"Starting continuous Kafka stream to {args.url} (~{args.rate:.1f} tx/s, bursts of {args.burst}). Press Ctrl+C to stop.")
        idx = 1
        sent = 0
        try:
            while True:
                wave = []
                for _ in range(args.burst):
                    if idx <= len(CURATED_SCENARIOS):
                        wave.append(CURATED_SCENARIOS[idx - 1])
                    else:
                        wave.append(generate_random_scenario(idx))
                    idx += 1
                sent += send_burst(args.url, wave, workers=min(args.burst, 8))
                time.sleep(max(0.2, args.burst / max(args.rate, 0.1)))
        except KeyboardInterrupt:
            print(f"\nStream stopped. Total queued: {sent} transactions.")
        return

    batch = list(CURATED_SCENARIOS)
    random.shuffle(batch)
    if args.count < len(batch):
        batch = batch[: args.count]
    elif args.count > len(batch):
        for i in range(args.count - len(batch)):
            batch.append(generate_random_scenario(i + 1))

    print(f"Streaming {len(batch)} transactions to {args.url} in waves of {args.burst} (creating live Kafka queue)...")
    sent = 0
    for start in range(0, len(batch), args.burst):
        wave = batch[start : start + args.burst]
        wave_num = (start // args.burst) + 1
        total_waves = (len(batch) + args.burst - 1) // args.burst
        print(f"\n--- Wave {wave_num}/{total_waves} ({len(wave)} concurrent events) ---", flush=True)
        sent += send_burst(args.url, wave, workers=min(len(wave), 8))
        if start + args.burst < len(batch) and args.wave_pause > 0:
            time.sleep(args.wave_pause)

    print(f"\nCompleted: {sent}/{len(batch)} transactions queued into Kafka.")
    print("Inspect live queue & results at http://localhost:3000")


if __name__ == "__main__":
    main()
