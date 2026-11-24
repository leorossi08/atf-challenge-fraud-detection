# ATF Fraud Detection Case — ML Solution, Kafka Event Pipeline & Operations Dashboard

**Repository:** https://github.com/leorossi08/atf-challenge-fraud-detection

End-to-end solution for the **Fraud Prevention Case (January/2026)**, combining:
1. **Data Science & Machine Learning (run remotely via `colab-cli` on a High-RAM Google Colab VM, with optional GPU):**
   - Exploration of 523,525 historical transactions from 2025 (`data/transacoes_historico_2025.csv`) and identification of the **4 spatio-temporal fraud regimes** that hold 100% of the 20,485 frauds.
   - Feature engineering (`services/ml-inference-service/train.py`) with constants fit on training data only, and **month-by-month rolling-origin (walk-forward) validation, Jul–Dec/2025** (263,913 out-of-time transactions) comparing 8 models. Thresholds and the champion are chosen from prior months only. Champion: **LightGBM** (**ROC-AUC = 0.985**, **PR-AUC = 0.687**); the tree models are statistically tied (see [`model-evaluation.md`](model-evaluation.md)).
   - Seasonality handling (December, Christmas proximity, holidays, vacation season) with an ablation and a December stress test.
   - Metrics (`models/fraud_models_cv.metrics.json`) and training constants (`models/domain_params.json`) are versioned; the pickled `.joblib` models are **not published** (see *Data and models*).
2. **Official Case Deliverables:**
   - **Single executed notebook:** [`Case_Fraude_Desenvolvimento.ipynb`](Case_Fraude_Desenvolvimento.ipynb) (code + analytical Markdown cells at every step + embedded charts).
   - `deliverables/transacoes_jan2026_classificadas.csv`: the January/2026 file (44,451 transactions) classified with `flag_fraude`, `probabilidade_fraude`, `flag_fraude_threshold_otimo` and `nivel_risco`.
   - `deliverables/Apresentacao_Case_Fraude_Jan2026.pdf`: executive presentation (widescreen 16:9).
3. **Architectural innovation — real-time detection system (Kafka + microservices + dashboard):**
   - **Operations Dashboard (`frontend/`):** React 19 + TypeScript + Nginx; monitors the transaction queue in real time, filters fraud alerts, inspects contextual attributes (merchant, city, device, score) and shows the *Model Card* with validation metrics.
   - **Transaction Gateway (`services/transaction-gateway/`):** Spring Boot 3 + JPA/PostgreSQL + Kafka Producer; validates case transactions, stores the `QUEUED` state and publishes to `transacoes-solicitadas`, partitioned by customer (`cardId`).
   - **Apache Kafka (`compose.yaml`):** KRaft broker isolated on the Compose private network with request (`transacoes-solicitadas`), result (`transacoes-analisadas`) and Dead-Letter (`transacoes-solicitadas.DLT`) topics.
   - **Fraud Analyzer (`services/fraud-analyzer/`):** Spring Boot + WebFlux + Kafka Consumer; consumes requests, calls the ML inference service with *retry* and DLT, and publishes `PROCESSING` and `COMPLETED` events.
   - **ML Inference Service (`services/ml-inference-service/`):** FastAPI that loads the `.joblib` pipeline you train on Colab (plus `domain_params.json`) and classifies transactions in `< 10 ms`.

---

## Key Case Insights

| Discovered Spatio-Temporal Regime | Time Window | Geographic Scope | Volume (2025) | Frauds (2025) | Fraud Rate | Rate w/ New Mobile |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **R1: Fuel stations (`Fuel`)** | 00h–08h | São Paulo and Rio de Janeiro | 5,495 | 2,633 | **47.92%** | **80.90%** |
| **R2: Food & Beverage (`Food & Beverage`)** | 12h–14h | South and Southeast (5 capitals) | 9,274 | 4,449 | **47.97%** | **80.65%** |
| **R3: Seasonal year-end e-commerce** | 20h–21h (Dec) | Nationwide (7 capitals) | 779 | 446 | **57.25%** | **83.72%** |
| **R4: Multi-merchant late-night window** | 22h–23h | Nationwide (all sectors) | 43,614 | 12,957 | **29.71%** | **65.47%** |
| **Safe zone (outside the 4 regimes)** | All other hours | Nationwide | 464,363 | 0 | **0.00%** | **0.00%** |

- **Legacy score failure:** `score_interno` (0–1000) has a correlation of `-0.0028` with actual fraud (mean `500.4` on legitimate vs `499.3` on frauds).
- **Seasonality:** real seasonality exists only in December (R3 appears, R4 rises from ≈ 29% to 37%, overall rate 3.7–3.9% → 5.42%). Calendar features gave **no measurable gain** (PR-AUC 0.6874 vs 0.6882 without), and the Jan–Nov model underestimates December by ≈ 0.9 pp; the Step-Up tier absorbs it. With only one year of data, a December effect and a holiday effect cannot be separated.
- **January/2026 classification (`44,451` transactions, unlabeled):**
  - **Expected volume (sum of probabilities):** `1,682.8` frauds (`3.79%`), in line with January/2025 (`3.73%`) — an estimate, not a validation.
  - **Automatic block (`p >= 0.50`):** `1,315` transactions flagged (`2.96%`), protecting **R$ 39,958.90** at **70.8% precision** and **0.93% false-positive rate** (out-of-time test).
  - **Step-Up Auth / 2FA policy (`p >= 0.288`, walk-forward F1 optimum):** `2,324` transactions monitored (`5.23%`), **R$ 69,400.10** protected, raising **Recall from 53.2% to 73.1%** (+2,133 frauds, +3,288 legitimate customers challenged).

---

## How to Run the Local Stack (Docker Compose)

```bash
cp .env.example .env
# Set a local password for POSTGRES_PASSWORD in the .env file
docker compose up --build
```

Services published on loopback only (`127.0.0.1`):

| Service | URL |
| --- | --- |
| Operations Dashboard (React) | http://localhost:3000 |
| Transaction Gateway API (Spring Boot) | http://localhost:8080 |
| ML Inference API Docs (FastAPI) | http://localhost:8000/docs |
| ML Inference Health | http://localhost:8000/health |
| Model Evaluation Metrics | http://localhost:8000/metrics |

To send real transactions from the January/2026 file through the Kafka queue:

```bash
python3 scripts/send_demo_transactions.py 25
bash scripts/smoke-test-flow.sh
```

---

## How to Reproduce the Training on Google Colab via `colab-cli`

All heavy ML work runs on Google Colab (`--high-mem`, T4 GPU by default for XGBoost/CatBoost); the resulting `.joblib` files are written to `models/` and are git-ignored. To re-run on Colab:

```bash
GPU=T4 bash scripts/colab_train_and_score.sh     # GPU=none for a CPU-only VM
```

Training runs as a background process on the VM and logs to `/content/train.log`; the script streams it. To follow it from a second terminal:

```bash
watch -n 10 'colab exec -s atf-fraud -f scripts/colab_tail.py --env N=30'
colab log -s atf-fraud -n 20
```

The protocol ablation that attributes the metric drop between the legacy and the current evaluation (`deliverables/ablation_protocol_drop.md`) is reproduced with `JOB=ablation bash scripts/colab_train_and_score.sh`.

To regenerate the figures, the executed notebook, the presentation `.pdf` and the `.zip` locally from the artifacts:

```bash
.venv/bin/python scripts/render_artefact_charts.py
.venv/bin/python scripts/render_architecture_diagrams.py
.venv/bin/python scripts/build_notebook.py
.venv/bin/python scripts/build_presentation_and_zip.py
```


## Data and models

- **Published:** the scored January/2026 CSV (`deliverables/`), the metrics JSONs and the executed notebook. The notebook's saved outputs are readable without running anything.
- **Not published, git-ignored `data/`:** the raw inputs `data/transacoes_historico_2025.csv` and `data/transacoes_jan2026.csv`. Place your own copies there to re-run the notebook or the training.
- **Not published:** the 9 `models/*.joblib` files (pickle executes code on load, so only load ones you trained yourself), `deliverables/oot_predictions.npz`, the delivery `.zip` bundles and the original case brief PDF.
- **Regenerate the models:** `GPU=none bash scripts/colab_train_and_score.sh` (see above). Until then the ML service starts without a model, and `scripts/build_notebook.py` stops with a message instead of failing mid-run.

## License

Code is released under the [MIT License](LICENSE). The data files, the case brief and the Artefact name and visual identity are **not** covered by that license.

## Automated Tests

```bash
.venv/bin/pytest services/ml-inference-service/test_inference.py
cd frontend && npm test && npm run build
```
