#!/usr/bin/env python3
"""
Generates two high-resolution (240 DPI) Artefact-branded architecture diagrams:
1. fig5_system_architecture.png — Event-Driven System Architecture (React Dashboard, Spring Boot
   Transaction Gateway, PostgreSQL, Apache Kafka KRaft topics + DLT, Fraud Analyzer, FastAPI ML Service,
   and Google Colab Remote Training Pipeline).
2. fig6_code_project_structure.png — Code & Project Architecture (Modules, Classes, Data Contracts,
   ML Pipeline Flow, and Deliverable Artifacts).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

BASE_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = BASE_DIR / "deliverables" / "figures"

NAVY = "#0D1634"
ROYAL = "#001994"
ELECTRIC = "#0046E1"
CYAN = "#1890D7"
MAGENTA = "#FF0066"
SOFT_PINK = "#FFF0F6"
ICE_BLUE = "#EEF3FF"
LIGHT_BG = "#F6F8FD"
WHITE = "#FFFFFF"
BORDER = "#CBD5E1"
MUTED = "#475569"


def configure_fonts():
    for fp in [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDisplay-Bold.ttf",
    ]:
        if Path(fp).is_file():
            font_manager.fontManager.addfont(fp)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Noto Sans", "DejaVu Sans"]


def draw_box(ax, x, y, w, h, title, lines, bg=WHITE, edge=ELECTRIC, title_bg=NAVY, title_fg=WHITE, lw=1.6):
    # Shadow
    ax.add_patch(FancyBboxPatch((x + 0.04, y - 0.05), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                facecolor="#D8E1F2", edgecolor="none", zorder=1))
    # Main card
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.14",
                                facecolor=bg, edgecolor=edge, linewidth=lw, zorder=2))
    # Header bar inside card
    hh = 0.42
    ax.add_patch(FancyBboxPatch((x, y + h - hh), w, hh, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=title_bg, edgecolor="none", zorder=3))
    ax.text(x + w / 2, y + h - hh / 2, title, ha="center", va="center",
            fontsize=9.3, fontweight="bold", color=title_fg, zorder=4)
    # Body lines
    body_text = "\n".join(lines)
    ax.text(x + 0.14, y + h - hh - 0.14, body_text, ha="left", va="top",
            fontsize=8.0, color=NAVY, linespacing=1.42, zorder=4)


def draw_arrow(ax, p1, p2, label="", color=ELECTRIC, rad=0.0, ls="-", label_offset=(0, 0.12)):
    arrow = FancyArrowPatch(
        p1, p2,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>,head_length=7,head_width=4.5",
        color=color,
        linewidth=1.8,
        linestyle=ls,
        zorder=5,
    )
    ax.add_patch(arrow)
    if label:
        mx = (p1[0] + p2[0]) / 2 + label_offset[0]
        my = (p1[1] + p2[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="center", fontsize=7.6, fontweight="bold",
                color=color, bbox=dict(boxstyle="round,pad=0.22", facecolor=WHITE, edgecolor=color, lw=0.8), zorder=6)


def render_system_architecture():
    fig, ax = plt.subplots(figsize=(14.5, 6.6), dpi=240)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Title banner
    ax.text(0.3, 6.85, "SYSTEM ARCHITECTURE — ASYNC KAFKA PIPELINE + ML INFERENCE + COLAB TRAINING",
            fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.3, 6.55, "End-to-end real-time flow (Docker Compose on 127.0.0.1) integrated with the remote colab-cli training pipeline",
            fontsize=9.0, color=MUTED)

    # Zone 1: Runtime Online Pipeline
    ax.add_patch(FancyBboxPatch((0.25, 2.25), 14.5, 4.10, boxstyle="round,pad=0.04,rounding_size=0.18",
                                facecolor="#FFFFFF", edgecolor="#CBD5E1", linewidth=1.1, linestyle="--", zorder=0))
    ax.text(0.45, 6.12, "ONLINE LAYER · DOCKER COMPOSE (PRIVATE NETWORK + LOOPBACK 127.0.0.1)",
            fontsize=8.2, fontweight="bold", color=ELECTRIC)

    draw_box(ax, 0.5, 3.85, 2.55, 2.0, "1. React Dashboard",
             ["• React 19 + TypeScript + Vite", "• Nginx Reverse Proxy (:3000)", "• Polling live queue (5s)", "• Filters & Record Inspector", "• Model Card (rolling-origin metrics)"],
             bg=WHITE, edge=ELECTRIC, title_bg=ELECTRIC)

    draw_box(ax, 3.6, 3.85, 2.75, 2.0, "2. Transaction Gateway",
             ["• Spring Boot 3.5 (Java 25)", "• POST /transactions (202)", "• GET /transactions (:8080)", "• Persists QUEUED state", "• Kafka Producer & Listener"],
             bg=WHITE, edge=ROYAL, title_bg=ROYAL)

    draw_box(ax, 7.05, 3.65, 2.8, 2.2, "3. Apache Kafka (KRaft)",
             ["• Private broker (:29092)", "• Topic 1: transacoes-solicitadas", "• Topic 2: transacoes-analisadas", "• Topic 3: ...solicitadas.DLT", "• Partition Key: cardId (customer)"],
             bg=ICE_BLUE, edge=NAVY, title_bg=NAVY)

    draw_box(ax, 10.45, 3.85, 2.05, 2.0, "4. Fraud Analyzer",
             ["• Spring Boot WebFlux", "• Consumes requests", "• Emits PROCESSING", "• Retry 3x + Backoff", "• Routes failures to DLT"],
             bg=WHITE, edge=ROYAL, title_bg=ROYAL)

    draw_box(ax, 12.85, 3.85, 1.75, 2.0, "5. FastAPI ML",
             ["• Uvicorn (:8000)", "• POST /predict", "• GET /metrics", "• Feature Eng.", "• Champion .joblib"],
             bg=SOFT_PINK, edge=MAGENTA, title_bg=MAGENTA)

    # PostgreSQL box below Gateway
    draw_box(ax, 3.75, 2.45, 2.45, 1.05, "PostgreSQL 16",
             ["• transactions table (:5432)", "• Lifecycle: QUEUED -> COMPLETED"],
             bg="#F8FAFC", edge=CYAN, title_bg=CYAN)

    # Arrows Online
    draw_arrow(ax, (3.07, 4.95), (3.58, 4.95), "/api/*", ELECTRIC, label_offset=(0, 0.22))
    draw_arrow(ax, (6.37, 5.15), (7.03, 5.15), "requested", ROYAL, label_offset=(0, 0.22))
    draw_arrow(ax, (7.03, 4.35), (6.37, 4.35), "analyzed / DLT", MAGENTA, label_offset=(0, -0.24))
    draw_arrow(ax, (9.87, 5.15), (10.43, 5.15), "consumes", NAVY, label_offset=(0, 0.22))
    draw_arrow(ax, (10.43, 4.35), (9.87, 4.35), "status", NAVY, label_offset=(0, -0.24))
    draw_arrow(ax, (12.52, 4.85), (12.83, 4.85), "HTTP", MAGENTA, label_offset=(0, 0.22))
    draw_arrow(ax, (4.97, 3.83), (4.97, 3.53), "JPA", CYAN, label_offset=(0.35, 0.0))
    # Dashboard to FastAPI metrics via Nginx
    draw_arrow(ax, (1.75, 3.83), (13.7, 3.83), "GET /api/model/metrics -> FastAPI /metrics", CYAN, rad=0.22, ls="--", label_offset=(0, -1.05))

    # Zone 2: Offline Colab ML Training & Artifact Delivery
    ax.add_patch(FancyBboxPatch((0.25, 0.20), 14.5, 1.85, boxstyle="round,pad=0.04,rounding_size=0.18",
                                facecolor="#0D1634", edgecolor=MAGENTA, linewidth=1.3, zorder=0))
    ax.text(0.45, 1.82, "OFFLINE LAYER · REMOTE TRAINING VIA COLAB-CLI (GOOGLE COLAB HIGH-RAM, OPTIONAL GPU) & PORTABLE ARTIFACTS",
            fontsize=8.2, fontweight="bold", color="#FF4D94")

    draw_box(ax, 0.55, 0.38, 3.2, 1.25, "Case Datasets (CSV)",
             ["• transacoes_historico_2025.csv (523k)", "• transacoes_jan2026.csv (44.4k)", "• Packaged by colab_train_and_score.sh"],
             bg=WHITE, edge=CYAN, title_bg=CYAN)

    draw_box(ax, 4.35, 0.38, 4.2, 1.25, "Google Colab High-RAM VM (colab-cli)",
             ["• Rolling-origin validation (Jul–Dec/2025)", "• Benchmarks 8 models, refits on all of 2025", "• Scores Jan/2026 and stops the VM (colab stop)"],
             bg=ICE_BLUE, edge=ELECTRIC, title_bg=ELECTRIC)

    draw_box(ax, 9.15, 0.38, 5.3, 1.25, "Artifacts Downloaded Locally (models/ & deliverables/)",
             ["• models/*.joblib + fraud_models_cv.metrics.json (consumed by FastAPI)", "• deliverables/transacoes_jan2026_classificadas.csv + PDF + .zip", "• Case_Fraude_Desenvolvimento.ipynb (executable notebook)"],
             bg=SOFT_PINK, edge=MAGENTA, title_bg=MAGENTA)

    draw_arrow(ax, (3.77, 1.0), (4.33, 1.0), "upload", WHITE, label_offset=(0, 0.24))
    draw_arrow(ax, (8.57, 1.0), (9.13, 1.0), "download", WHITE, label_offset=(0, 0.24))
    draw_arrow(ax, (13.7, 1.65), (13.7, 3.83), "mount .joblib", MAGENTA, ls=":", label_offset=(-0.75, 0.0))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_system_architecture.png", bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close(fig)


def render_code_project_architecture():
    fig, ax = plt.subplots(figsize=(14.5, 6.6), dpi=240)
    fig.patch.set_facecolor(LIGHT_BG)
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    ax.text(0.3, 6.85, "CODE ARCHITECTURE AND REPOSITORY LAYOUT",
            fontsize=12.5, fontweight="bold", color=NAVY)
    ax.text(0.3, 6.55, "Modules, data contracts, main classes and automation scripts in atf-challenge-fraud-detection",
            fontsize=9.0, color=MUTED)

    # Top row: 4 core code modules
    draw_box(ax, 0.35, 3.55, 3.35, 2.75, "services/ml-inference-service/",
             ["train.py:", "  • engineer_features() (36 num + 7 cat)", "  • build_models() (8 models, rolling-origin)", "  • walk-forward selection & bootstrap_ci()", "main.py (FastAPI):", "  • PredictionRequest (case schema)", "  • POST /predict | GET /metrics", "test_inference.py (7 pytest tests)", "Dockerfile (python:3.12 + libgomp1)"],
             bg=SOFT_PINK, edge=MAGENTA, title_bg=MAGENTA)

    draw_box(ax, 3.95, 3.55, 3.45, 2.75, "services/transaction-gateway/",
             ["controller/TransactionController.java", "  • POST /transactions -> 202 Accepted", "  • GET /transactions?limit=100", "dto/TransactionDTO.java (record)", "  • Business fields from the case", "model/TransactionRecord.java (JPA)", "service/TransactionLifecycleService", "service/KafkaProducerService.java", "listener/TransactionStatusListener"],
             bg=WHITE, edge=ROYAL, title_bg=ROYAL)

    draw_box(ax, 7.65, 3.55, 3.35, 2.75, "services/fraud-analyzer/",
             ["FraudAnalyzerApplication.java", "  • Declares Kafka topics + DLT", "listener/FraudListener.java", "  • @KafkaListener(solicitadas)", "  • WebClient -> FastAPI /predict", "  • Publica PROCESSING / COMPLETED", "config/KafkaConsumerConfiguration", "  • DefaultErrorHandler (3 retries)", "  • DeadLetterPublishingRecoverer"],
             bg=WHITE, edge=ELECTRIC, title_bg=ELECTRIC)

    draw_box(ax, 11.25, 3.55, 3.40, 2.75, "frontend/ (React + Vite)",
             ["src/App.tsx:", "  • Live KPIs & filter bar", "  • Table with merchant, city, device", "  • TransactionDetail (Record Inspector)", "  • Jan/2026 simulation modal", "  • Model Card (rolling-origin metrics)", "src/App.test.tsx (3 Vitest tests)", "nginx.conf (/api/ -> Gateway & ML)", "Dockerfile (node:22 -> nginx:1.27)"],
             bg=ICE_BLUE, edge=NAVY, title_bg=NAVY)

    # Bottom row: Automation, Artifacts & Deliverables
    draw_box(ax, 0.35, 0.45, 4.5, 2.65, "scripts/ (Automation & Reproducibility)",
             ["• colab_train_and_score.sh: provisions a Colab VM (GPU optional), trains, downloads .joblib", "• render_artefact_charts.py: renders 240-DPI PNG charts (Artefact palette)", "• render_architecture_diagrams.py: renders system and code diagrams", "• build_notebook.py: builds and runs Case_Fraude_Desenvolvimento.ipynb", "• build_presentation_and_zip.py: builds the executive PDF + delivery .zip", "• send_demo_transactions.py: injects a real Jan/2026 batch into Kafka", "• smoke-test-flow.sh: end-to-end test HTTP -> Kafka -> PostgreSQL"],
             bg=WHITE, edge=CYAN, title_bg=CYAN)

    draw_box(ax, 5.15, 0.45, 4.45, 2.65, "models/ & data/ (Artifacts & Data)",
             ["models/ (exported from Google Colab):", "  • champion_fraud_model_cv.joblib (active model)", "  • lightgbm_fraud_model_cv.joblib", "  • logistic_regression_fraud_model_cv.joblib", "  • fraud_models_cv.metrics.json (rolling-origin, CIs)", "data/:", "  • transacoes_historico_2025.csv (git-ignored)", "  • transacoes_jan2026.csv (44,451 transactions)"],
             bg=WHITE, edge=ELECTRIC, title_bg=ELECTRIC)

    draw_box(ax, 9.90, 0.45, 4.75, 2.65, "deliverables/ & Root (Case Deliverables)",
             ["• Case_Fraude_Desenvolvimento.ipynb (single notebook)", "• Entrega_Case_Fraude_Jan2026.zip (official package)", "• deliverables/transacoes_jan2026_classificadas.csv", "• deliverables/Apresentacao_Case_Fraude_Jan2026.pdf", "• deliverables/figures/fig1..fig6.png", "• compose.yaml (6-container orchestration)", "• README.md & run.md (technical documentation)"],
             bg=SOFT_PINK, edge=MAGENTA, title_bg=MAGENTA)

    # Connecting arrows showing code dependencies
    draw_arrow(ax, (11.23, 5.0), (7.42, 5.0), "", ELECTRIC, rad=-0.18, ls="--")
    draw_arrow(ax, (7.42, 4.45), (7.63, 4.45), "", ROYAL)
    draw_arrow(ax, (3.72, 4.45), (7.63, 4.45), "", MAGENTA, rad=0.14, ls=":")
    draw_arrow(ax, (2.6, 3.12), (2.0, 3.53), "builds/tests", CYAN, label_offset=(-0.55, 0.0))
    draw_arrow(ax, (7.3, 3.12), (2.2, 3.53), "loads .joblib", MAGENTA, rad=0.08, ls="--", label_offset=(0.2, 0.12))
    draw_arrow(ax, (9.62, 1.8), (9.88, 1.8), "", MAGENTA)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_code_project_structure.png", bbox_inches="tight", facecolor=LIGHT_BG)
    plt.close(fig)


if __name__ == "__main__":
    configure_fonts()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    render_system_architecture()
    render_code_project_architecture()
    print("Architecture diagrams generated: fig5_system_architecture.png & fig6_code_project_structure.png")
