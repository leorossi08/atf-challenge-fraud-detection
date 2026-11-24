# System and Code Architecture — ATF Fraud Detection Case

**Repository:** https://github.com/leorossi08/atf-challenge-fraud-detection

This document presents the architecture diagrams of the event-driven system (Kafka + Microservices + FastAPI + React) and the structure of the code and the remote training flow on Google Colab (`colab-cli`).

---

## 1. System Architecture (Real Time + Remote Colab Training)

![System Architecture](deliverables/figures/fig5_system_architecture.png)

```mermaid
flowchart LR
    subgraph ONLINE["Online Layer (Docker Compose · 127.0.0.1)"]
        UI["React Dashboard\n(Nginx :3000)"]
        GW["Transaction Gateway\n(Spring Boot :8080)"]
        DB[("PostgreSQL 16\ntransactions :5432")]
        K1[["Kafka Topic\ntransacoes-solicitadas"]]
        K2[["Kafka Topic\ntransacoes-analisadas"]]
        K3[["Kafka DLT\ntransacoes-solicitadas.DLT"]]
        FA["Fraud Analyzer\n(Spring Boot WebFlux)"]
        ML["FastAPI ML Inference\n(Uvicorn :8000)"]

        UI -->|POST /api/transactions\nGET /api/transactions| GW
        UI -.->|GET /api/model/metrics| ML
        GW -->|JPA QUEUED/COMPLETED| DB
        GW -->|Key = cardId| K1
        K1 -->|Consume| FA
        FA -->|POST /predict| ML
        FA -->|PROCESSING / COMPLETED| K2
        FA -.->|After 3 retries| K3
        K2 -->|Update Status| GW
        K3 -->|Mark FAILED| GW
    end

    subgraph OFFLINE["Offline Layer (Google Colab High-RAM, optional GPU, via colab-cli)"]
        DATA["CSV 2025 (523k)\nCSV Jan/2026 (44.4k)"]
        COLAB["Colab VM High-RAM (GPU optional)\nRolling-origin validation + 8 models"]
        ART["models/*.joblib\nfraud_models_cv.metrics.json\nClassified Jan-2026"]

        DATA -->|colab upload| COLAB
        COLAB -->|colab download + colab stop| ART
    end

    ART -.->|Mount .joblib| ML
```

---

## 2. Transaction Lifecycle (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    participant User as Dashboard / Script Jan-2026
    participant GW as Transaction Gateway (Spring Boot)
    participant PG as PostgreSQL
    participant K as Apache Kafka (KRaft)
    participant FA as Fraud Analyzer (WebFlux)
    participant ML as FastAPI ML (.joblib)

    User->>GW: POST /transactions (case payload)
    GW->>PG: INSERT status = QUEUED
    GW->>K: Produce -> transacoes-solicitadas (key = nome_cliente)
    GW-->>User: 202 Accepted {transactionId, status: QUEUED}

    K->>FA: Consume transaction event
    FA->>K: Produce -> transacoes-analisadas (status: PROCESSING)
    K->>GW: Update status -> PROCESSING
    FA->>ML: POST /predict (engineer_features + champion .joblib)
    ML-->>FA: {isFraud, fraudProbability}
    FA->>K: Produce -> transacoes-analisadas (COMPLETED, isFraud, prob)
    K->>GW: Update status -> COMPLETED in PostgreSQL
    User->>GW: GET /transactions?limit=100 (polling 5s)
    GW-->>User: Show decision, score and inspector in the Dashboard
```

---

## 3. Code Architecture and Project Structure

![Code Architecture](deliverables/figures/fig6_code_project_structure.png)

```mermaid
graph TD
    ROOT["atf-fraud-detection-case/"]
    ROOT --> NB["Case_Fraude_Desenvolvimento.ipynb\n(Single executable notebook)"]
    ROOT --> CMP["compose.yaml\n(6 containers: Kafka, PG, GW, FA, ML, UI)"]

    ROOT --> SVC["services/"]
    SVC --> ML["ml-inference-service/\n• train.py (feature eng + rolling-origin validation)\n• main.py (FastAPI /predict, /metrics)\n• test_inference.py (10 pytest tests)"]
    SVC --> GW["transaction-gateway/\n• TransactionController.java\n• TransactionDTO.java & TransactionRecord.java\n• TransactionLifecycleService.java\n• KafkaProducerService & StatusListener"]
    SVC --> FA["fraud-analyzer/\n• FraudListener.java (WebClient + Kafka)\n• KafkaConsumerConfiguration.java (Retry + DLT)"]

    ROOT --> FE["frontend/\n• src/App.tsx (React Dashboard + Model Card)\n• src/App.test.tsx (3 Vitest tests)\n• nginx.conf (Reverse Proxy)"]

    ROOT --> SCR["scripts/\n• colab_train_and_score.sh\n• render_artefact_charts.py\n• render_architecture_diagrams.py\n• build_notebook.py\n• build_presentation_and_zip.py\n• send_demo_transactions.py & smoke-test-flow.sh"]

    ROOT --> MOD["models/\n• champion_fraud_model_cv.joblib (+ 8 model artifacts)\n• domain_params.json\n• fraud_models_cv.metrics.json"]

    ROOT --> DEL["deliverables/\n• transacoes_jan2026_classificadas.csv\n• Apresentacao_Case_Fraude_Jan2026.pdf\n• figures/fig1..fig6.png"]
```
