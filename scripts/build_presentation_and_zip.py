#!/usr/bin/env python3
"""
Generates a modern 8-slide widescreen (16:9) Executive Presentation PDF for the Artefact (ATF)
Fraud Detection Case:
- Modern geometric typography via embedded NotoSansDisplay & NotoSans TrueType fonts
- Full dark Midnight Navy (#0D1634) executive cover slide with geometric Artefact data-grid accents
- Subtle geometric consulting backgrounds, layered drop shadows, rounded KPI pills, and high-contrast tables
- Embeds the freshly rendered Artefact-branded 240-DPI matplotlib figures
- Repackages the official deliverable .zip archive.
"""
import json
import zipfile

import pandas as pd
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

BASE_DIR = Path(__file__).resolve().parents[1]
DELIVERABLES_DIR = BASE_DIR / "deliverables"
FIGURES_DIR = DELIVERABLES_DIR / "figures"
METRICS_PATH = BASE_DIR / "models" / "fraud_models_cv.metrics.json"
PDF_PATH = DELIVERABLES_DIR / "Apresentacao_Case_Fraude_Jan2026.pdf"
ZIP_PATH = BASE_DIR / "Entrega_Case_Fraude_Jan2026.zip"
ZIP_DELIVERABLE_COPY = DELIVERABLES_DIR / "Entrega_Case_Fraude_Jan2026.zip"

SLIDE_W = 13.333 * inch
SLIDE_H = 7.5 * inch

# Artefact Brand Palette
NAVY = colors.HexColor("#0D1634")
NAVY_MID = colors.HexColor("#162350")
ROYAL = colors.HexColor("#001994")
ELECTRIC = colors.HexColor("#0046E1")
CYAN = colors.HexColor("#1890D7")
MAGENTA = colors.HexColor("#FF0066")
SOFT_PINK = colors.HexColor("#FFF0F6")
ICE_BLUE = colors.HexColor("#EEF3FF")
CANVAS_BG = colors.HexColor("#F3F6FC")
PAPER = colors.HexColor("#FFFFFF")
INK = colors.HexColor("#0F172A")
MUTED = colors.HexColor("#52607D")
BORDER = colors.HexColor("#DCE4F4")
SHADOW = colors.HexColor("#DFE6F5")

# Register modern Noto Sans / Noto Sans Display fonts
FONT_REG = "NotoSans"
FONT_MED = "NotoSansMed"
FONT_SEMI = "NotoSansSemi"
FONT_BOLD = "NotoSansBold"
FONT_DISP = "NotoDisplayBold"
FONT_DISP_BLACK = "NotoDisplayBlack"


def register_fonts():
    font_map = {
        FONT_REG: "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        FONT_MED: "/usr/share/fonts/truetype/noto/NotoSans-Medium.ttf",
        FONT_SEMI: "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
        FONT_BOLD: "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        FONT_DISP: "/usr/share/fonts/truetype/noto/NotoSansDisplay-Bold.ttf",
        FONT_DISP_BLACK: "/usr/share/fonts/truetype/noto/NotoSansDisplay-ExtraBold.ttf",
    }
    for alias, path in font_map.items():
        if Path(path).is_file():
            pdfmetrics.registerFont(TTFont(alias, path))
    pdfmetrics.registerFontFamily(
        FONT_REG,
        normal=FONT_REG,
        bold=FONT_BOLD,
        italic=FONT_REG,
        boldItalic=FONT_BOLD,
    )


def draw_artefact_mark(c: canvas.Canvas, x, y, dark_bg=True, scale=1.0):
    c.saveState()
    c.setFont(FONT_DISP_BLACK, 12.5 * scale)
    c.setFillColor(PAPER if dark_bg else NAVY)
    c.drawString(x, y, "ARTEFACT")
    c.setFillColor(MAGENTA)
    c.roundRect(x + 1.15 * inch * scale, y + 0.01 * inch * scale, 0.10 * inch * scale, 0.10 * inch * scale, 2, stroke=0, fill=1)
    c.setFont(FONT_SEMI, 7.5 * scale)
    c.setFillColor(colors.HexColor("#8FA3D6") if dark_bg else MUTED)
    c.drawString(x + 1.33 * inch * scale, y + 0.02 * inch * scale, "DATA & AI CASE")
    c.restoreState()


def draw_modern_content_bg(c: canvas.Canvas, title: str, kicker: str, page_num: int, total_pages: int = 9):
    # Cool ice-gray background
    c.setFillColor(CANVAS_BG)
    c.rect(0, 0, SLIDE_W, SLIDE_H, stroke=0, fill=1)

    # Subtle modern geometric corner glow / grid accents
    c.saveState()
    c.setFillColor(colors.HexColor("#E7EEFA"))
    c.circle(SLIDE_W - 0.6 * inch, 0.9 * inch, 2.3 * inch, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#FCECF3"))
    c.circle(0.5 * inch, 0.4 * inch, 1.3 * inch, stroke=0, fill=1)
    # Subtle dot matrix top right below header
    c.setFillColor(colors.HexColor("#D2DCF0"))
    for gx in range(8):
        for gy in range(3):
            c.circle(SLIDE_W - 0.45 * inch - gx * 0.16 * inch, SLIDE_H - 1.18 * inch - gy * 0.16 * inch, 1.1, stroke=0, fill=1)
    c.restoreState()

    # Sleek Dark Navy Header Bar
    c.setFillColor(NAVY)
    c.rect(0, SLIDE_H - 0.96 * inch, SLIDE_W, 0.96 * inch, stroke=0, fill=1)

    # Dynamic Magenta + Electric Blue Gradient-like Accent Line
    c.setFillColor(MAGENTA)
    c.rect(0, SLIDE_H - 0.96 * inch, 0.13 * inch, 0.96 * inch, stroke=0, fill=1)
    c.rect(0, SLIDE_H - 1.00 * inch, 3.4 * inch, 0.04 * inch, stroke=0, fill=1)
    c.setFillColor(ELECTRIC)
    c.rect(3.4 * inch, SLIDE_H - 1.00 * inch, 6.2 * inch, 0.04 * inch, stroke=0, fill=1)
    c.setFillColor(CYAN)
    c.rect(9.6 * inch, SLIDE_H - 1.00 * inch, SLIDE_W - 9.6 * inch, 0.04 * inch, stroke=0, fill=1)

    # Kicker pill
    c.setFillColor(colors.HexColor("#FF4D94"))
    c.setFont(FONT_BOLD, 8.2)
    c.drawString(0.55 * inch, SLIDE_H - 0.34 * inch, kicker.upper())

    # Slide Title in Noto Sans Display
    c.setFillColor(PAPER)
    c.setFont(FONT_DISP, 18.5)
    c.drawString(0.55 * inch, SLIDE_H - 0.70 * inch, title)

    draw_artefact_mark(c, SLIDE_W - 2.65 * inch, SLIDE_H - 0.56 * inch, dark_bg=True)

    # Minimalist Footer
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.8)
    c.line(0.55 * inch, 0.44 * inch, SLIDE_W - 0.55 * inch, 0.44 * inch)
    c.setFillColor(MUTED)
    c.setFont(FONT_REG, 8.2)
    c.drawString(0.55 * inch, 0.21 * inch, "Fraud Prevention Case (Artefact)  ·  Machine Learning, Apache Kafka & Real-Time Operations Dashboard")
    c.setFillColor(NAVY)
    c.setFont(FONT_BOLD, 8.5)
    c.drawRightString(SLIDE_W - 0.55 * inch, 0.21 * inch, f"{page_num:02d} / {total_pages:02d}")


def draw_card(c: canvas.Canvas, x, y, w, h, title=None, bg=PAPER, border_color=BORDER, accent_bar=None):
    # Soft elevation shadow
    c.setFillColor(SHADOW)
    c.roundRect(x + 2, y - 2.5, w, h, 9, stroke=0, fill=1)
    # Main card
    c.setFillColor(bg)
    c.setStrokeColor(border_color)
    c.setLineWidth(0.9)
    c.roundRect(x, y, w, h, 9, stroke=1, fill=1)
    if accent_bar:
        c.setFillColor(accent_bar)
        c.roundRect(x, y + h - 0.065 * inch, w, 0.065 * inch, 3, stroke=0, fill=1)
    if title:
        c.setFillColor(ROYAL)
        c.setFont(FONT_DISP, 11)
        c.drawString(x + 0.20 * inch, y + h - 0.34 * inch, title)


def draw_kpi_box(c: canvas.Canvas, x, y, w, h, label, value, sub, highlight=False):
    draw_card(
        c,
        x,
        y,
        w,
        h,
        bg=SOFT_PINK if highlight else PAPER,
        border_color=MAGENTA if highlight else BORDER,
        accent_bar=MAGENTA if highlight else ELECTRIC,
    )
    c.setFillColor(MUTED)
    c.setFont(FONT_SEMI, 8.4)
    c.drawString(x + 0.20 * inch, y + h - 0.31 * inch, label.upper())

    c.setFillColor(MAGENTA if highlight else NAVY)
    c.setFont(FONT_DISP_BLACK, 22)
    c.drawString(x + 0.20 * inch, y + h - 0.72 * inch, value)

    c.setFillColor(INK)
    c.setFont(FONT_REG, 8.8)
    c.drawString(x + 0.20 * inch, y + 0.18 * inch, sub)


def draw_cover_slide(c: canvas.Canvas, jan: dict):
    # Full Dark Midnight Navy Background
    c.setFillColor(NAVY)
    c.rect(0, 0, SLIDE_W, SLIDE_H, stroke=0, fill=1)

    # Geometric Modern Data-Consulting Background Shapes
    c.saveState()
    c.setFillColor(NAVY_MID)
    c.circle(SLIDE_W - 1.2 * inch, SLIDE_H - 1.5 * inch, 4.2 * inch, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#1B2B63"))
    c.circle(SLIDE_W - 0.4 * inch, 1.0 * inch, 2.8 * inch, stroke=0, fill=1)

    # Subtle glowing accent lines
    c.setStrokeColor(ELECTRIC)
    c.setLineWidth(1.5)
    c.line(0, 0.95 * inch, SLIDE_W, 0.95 * inch)
    c.setFillColor(MAGENTA)
    c.rect(0, 0, 0.22 * inch, SLIDE_H, stroke=0, fill=1)
    c.rect(0.85 * inch, 4.45 * inch, 1.15 * inch, 0.07 * inch, stroke=0, fill=1)

    # Dot grid accent right side
    c.setFillColor(colors.HexColor("#2D4285"))
    for gx in range(10):
        for gy in range(8):
            c.circle(SLIDE_W - 0.8 * inch - gx * 0.25 * inch, SLIDE_H - 1.0 * inch - gy * 0.25 * inch, 1.5, stroke=0, fill=1)
    c.restoreState()

    # Top Wordmark
    draw_artefact_mark(c, 0.85 * inch, SLIDE_H - 0.95 * inch, dark_bg=True, scale=1.25)

    # Pill Tag
    c.setFillColor(colors.HexColor("#23336B"))
    c.roundRect(0.85 * inch, 5.65 * inch, 3.55 * inch, 0.34 * inch, 12, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#FF4D94"))
    c.setFont(FONT_BOLD, 8.8)
    c.drawString(1.03 * inch, 5.77 * inch, "DATA SCIENCE CASE  ·  FRAUD PREVENTION")

    # Hero Title
    c.setFillColor(PAPER)
    c.setFont(FONT_DISP_BLACK, 34)
    c.drawString(0.85 * inch, 5.05 * inch, "Real-Time Fraud Detection")
    c.setFont(FONT_DISP, 24)
    c.setFillColor(colors.HexColor("#BFD3FF"))
    c.drawString(0.85 * inch, 4.62 * inch, "Predictive Modeling, Kafka Streaming & January/2026 Operations")

    # Subtitle
    c.setFillColor(colors.HexColor("#CBD5E1"))
    c.setFont(FONT_REG, 11.5)
    c.drawString(0.85 * inch, 3.92 * inch, "End-to-end solution combining feature engineering over 523,525 historical transactions,")
    c.drawString(0.85 * inch, 3.66 * inch, "a dual-threshold decision policy and an async Spring Boot + Kafka + FastAPI + React architecture.")

    # 3 Hero Metric Badges on Cover
    cover_kpis = [
        ("0.9848", "Out-of-Time ROC-AUC (Jul–Dec)", ELECTRIC),
        ("100%", "Of frauds mapped to 4 regimes", MAGENTA),
        (f"{jan['flagged_at_0_50']:,} / {jan['flagged_at_optimal_f1']:,}", "Frauds flagged Jan/2026", CYAN),
    ]
    for idx, (val, lbl, col) in enumerate(cover_kpis):
        bx = 0.85 * inch + idx * 3.85 * inch
        by = 1.65 * inch
        c.setFillColor(colors.HexColor("#14204A"))
        c.setStrokeColor(col)
        c.setLineWidth(1.2)
        c.roundRect(bx, by, 3.55 * inch, 1.35 * inch, 10, stroke=1, fill=1)
        c.setFillColor(col)
        c.rect(bx, by + 1.27 * inch, 3.55 * inch, 0.08 * inch, stroke=0, fill=1)
        c.setFillColor(PAPER)
        c.setFont(FONT_DISP_BLACK, 24)
        c.drawString(bx + 0.24 * inch, by + 0.65 * inch, val)
        c.setFillColor(colors.HexColor("#A9BBE0"))
        c.setFont(FONT_MED, 9.8)
        c.drawString(bx + 0.24 * inch, by + 0.28 * inch, lbl)

    # Bottom Metadata
    c.setFillColor(colors.HexColor("#8FA3D6"))
    c.setFont(FONT_REG, 9.2)
    c.drawString(0.85 * inch, 0.42 * inch, "Leonardo Rossi  ·  https://github.com/leorossi08/atf-challenge-fraud-detection")
    c.setFillColor(PAPER)
    c.setFont(FONT_BOLD, 9.2)
    c.drawRightString(SLIDE_W - 0.65 * inch, 0.42 * inch, "01 / 09")
    c.showPage()


def build_pdf():
    register_fonts()
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    jan = report["january_2026_forecast"]
    jan_scored = pd.read_csv(BASE_DIR / "deliverables" / "transacoes_jan2026_classificadas.csv")

    def sector_rate(name: str) -> float:
        return float(jan_scored.loc[jan_scored["tipo_estabelecimento"] == name, "flag_fraude"].mean() * 100.0)

    xgb_05 = report["test_metrics_at_threshold_0_5"]["xgboost"]
    xgb_opt = report["test_metrics_at_optimal_f1_threshold"]["xgboost"]
    lgb_05 = report["test_metrics_at_threshold_0_5"]["lightgbm"]
    lr_05 = report["test_metrics_at_threshold_0_5"]["logistic_regression"]

    DELIVERABLES_DIR.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(PDF_PATH), pagesize=landscape((SLIDE_W, SLIDE_H)))

    styles = getSampleStyleSheet()
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName=FONT_REG,
        fontSize=9.8,
        leading=14.0,
        textColor=INK,
        spaceAfter=5,
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        parent=body_style,
        leftIndent=8,
        spaceAfter=5,
    )

    # ================= SLIDE 1: MODERN EXECUTIVE COVER =================
    draw_cover_slide(c, jan)

    # ================= SLIDE 2: EXECUTIVE SUMMARY =================
    draw_modern_content_bg(
        c,
        "Executive Summary: The Challenge and the Proposed Solution",
        "Overview · Business Context and Expected Impact",
        2,
    )
    draw_kpi_box(c, 0.55 * inch, 5.05 * inch, 2.9 * inch, 1.22 * inch, "2025 Historical Data", "523,525", "20,485 frauds (3.91% of the base)")
    draw_kpi_box(c, 3.65 * inch, 5.05 * inch, 2.9 * inch, 1.22 * inch, "Out-of-Time Performance (Jul–Dec)", f"{lgb_05['roc_auc']:.4f} AUC", f"PR-AUC: {lgb_05['average_precision']:.4f} (~17x over baseline)")
    draw_kpi_box(c, 6.75 * inch, 5.05 * inch, 2.9 * inch, 1.22 * inch, "January/2026 Data", "44,451", "100% classified with score and tier")
    draw_kpi_box(c, 9.85 * inch, 5.05 * inch, 2.9 * inch, 1.22 * inch, "Frauds Flagged Jan/26", f"{jan['flagged_at_0_50']:,} / {jan['flagged_at_optimal_f1']:,}", f"R$ {jan['amount_flagged_at_0_50_brl']/1000:.1f}k (@0.50) to R$ {jan['amount_flagged_at_optimal_brl']/1000:.1f}k (@{jan['optimal_f1_threshold']:.3f})", highlight=True)

    draw_card(c, 0.55 * inch, 0.70 * inch, 5.95 * inch, 4.10 * inch, "1. The Business Challenge", accent_bar=NAVY)
    p1 = Paragraph(
        "• <b>Scale vs. Precision:</b> The current manual process is a bottleneck against a volume of ~44.5k transactions/month, causing losses from frauds not detected in time.<br/><br/>"
        "• <b>Legitimate Customer Experience:</b> Poorly calibrated preventive blocks create severe friction and lost revenue on good customers.<br/><br/>"
        "• <b>Legacy Score Failure:</b> Our analysis showed that the current <i>score_interno</i> (0–1000) has essentially zero correlation (-0.0028) with actual fraud.<br/><br/>"
        "• <b>Our Solution:</b> A Gradient Boosting model + an async messaging pipeline (Apache Kafka + Spring Boot + FastAPI + React Dashboard) with tiered decisions.",
        bullet_style,
    )
    p1.wrapOn(c, 5.55 * inch, 3.3 * inch)
    p1.drawOn(c, 0.75 * inch, 1.25 * inch)

    draw_card(c, 6.80 * inch, 0.70 * inch, 5.95 * inch, 4.10 * inch, "2. Key Deliverables and Technical Highlights", accent_bar=MAGENTA)
    p2 = Paragraph(
        "• <b>Discovery of the 4 Spatio-Temporal Regimes:</b> 100% of historical frauds occur in 4 specific time/region windows, strongly amplified by new mobile devices (risk up to 83.7%).<br/><br/>"
        "• <b>Rigorous Out-of-Time Validation (Colab, GPU optional):</b> month-by-month rolling-origin validation (Jul–Dec/2025, 263,913 transactions); thresholds and champion chosen using prior months only.<br/><br/>"
        "• <b>Dual-Threshold Policy:</b> automatic block at p >= 0.50 (70.8% precision, <1% false positives) and Step-Up Auth (2FA) at 0.288 <= p < 0.50 (raising Recall to 73.1%).<br/><br/>"
        "• <b>Production-Ready Full-Stack Platform:</b> Microservices decoupled via Kafka with a Dead-Letter Topic (DLT) and a real-time operations dashboard.",
        bullet_style,
    )
    p2.wrapOn(c, 5.55 * inch, 3.3 * inch)
    p2.drawOn(c, 7.00 * inch, 1.08 * inch)
    c.showPage()

    # ================= SLIDE 3: EDA & 4 SPATIO-TEMPORAL REGIMES =================
    draw_modern_content_bg(
        c,
        "Exploratory Diagnosis: The 4 Spatio-Temporal Fraud Regimes",
        "Exploratory Data Analysis (EDA) · 2025 History (523,525 Transactions)",
        3,
    )
    draw_card(c, 0.55 * inch, 2.72 * inch, 12.2 * inch, 3.58 * inch)
    fig1 = FIGURES_DIR / "fig1_eda_regimes_device.png"
    if fig1.is_file():
        c.drawImage(str(fig1), 0.70 * inch, 2.80 * inch, width=11.9 * inch, height=3.42 * inch, preserveAspectRatio=True)

    draw_card(c, 0.55 * inch, 0.62 * inch, 12.2 * inch, 1.92 * inch, "Summary of the 4 Critical Regimes Found in the 2025 Data", accent_bar=ELECTRIC)
    table_data = [
        ["Regime", "Time Window", "Geographic Scope", "2025 Volume", "2025 Frauds", "Fraud Rate", "Rate w/ New Mobile"],
        ["R1: Fuel Stations", "00h–08h", "São Paulo and Rio de Janeiro", "5,495", "2,633", "47.92%", "80.90%"],
        ["R2: Food & Beverage", "12h–14h", "South and Southeast (5 capitals)", "9,274", "4,449", "47.97%", "80.65%"],
        ["R3: Seasonal Christmas E-commerce", "20h–21h (Dec)", "Nationwide (7 capitals)", "779", "446", "57.25%", "83.72%"],
        ["R4: Multi-Sector Late Night", "22h–23h", "Nationwide (all sectors)", "43,614", "12,957", "29.71%", "65.47%"],
    ]
    t = Table(table_data, colWidths=[2.5 * inch, 1.3 * inch, 2.2 * inch, 1.2 * inch, 1.2 * inch, 1.4 * inch, 1.8 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), PAPER),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT_REG),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("ALIGN", (3, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, ICE_BLUE]),
                ("TEXTCOLOR", (6, 1), (6, -1), MAGENTA),
                ("FONTNAME", (6, 1), (6, -1), FONT_BOLD),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    t.wrapOn(c, 11.8 * inch, 1.4 * inch)
    t.drawOn(c, 0.75 * inch, 0.74 * inch)
    c.showPage()

    # ================= SLIDE 4: MODEL BENCHMARK =================
    draw_modern_content_bg(
        c,
        "Predictive Modeling and Out-of-Time Validation (Rolling-Origin, Jul–Dec/2025)",
        "Feature Engineering · Training on Google Colab High-RAM (GPU optional)",
        4,
    )
    draw_card(c, 0.55 * inch, 2.72 * inch, 12.2 * inch, 3.58 * inch)
    fig2 = FIGURES_DIR / "fig2_roc_pr_curves.png"
    if fig2.is_file():
        c.drawImage(str(fig2), 0.70 * inch, 2.80 * inch, width=11.9 * inch, height=3.42 * inch, preserveAspectRatio=True)

    draw_card(c, 0.55 * inch, 0.62 * inch, 12.2 * inch, 1.96 * inch, "8-Model Benchmark: Pooled Out-of-Time (Jul–Dec), Nov–Dec, and Latency", accent_bar=ELECTRIC)
    bench_order = [
        ("lightgbm", "LightGBM ★ (Champion)"),
        ("soft_voting_ensemble", "Soft-Voting Ensemble"),
        ("xgboost", "XGBoost (Regularized)"),
        ("catboost", "CatBoost"),
        ("hist_gradient_boosting", "HistGradientBoosting"),
        ("random_forest", "Random Forest"),
        ("logistic_regression", "Logistic Regression"),
    ]
    m_data = [["Model", "Pooled PR-AUC", "Nov–Dec PR-AUC", "ROC-AUC", "Brier", "F1 @0.50", "F1 @Walk-fwd thr.", "p50 (ms)"]]
    for key, label in bench_order:
        t05 = report["test_metrics_at_threshold_0_5"][key]
        topt = report["test_metrics_at_optimal_f1_threshold"][key]
        ch = report["chronological_validation_nov_dec"][key]
        ops = report["operational_tradeoffs"][key]
        m_data.append([
            label,
            f"{t05['average_precision']:.4f}",
            f"{ch['average_precision']:.4f}",
            f"{t05['roc_auc']:.4f}",
            f"{t05['brier_score']:.5f}",
            f"{t05['f1']:.4f}",
            f"{topt['f1']:.4f}",
            f"{ops['p50_ms']:.2f} ms",
        ])
    tm = Table(m_data, colWidths=[2.3 * inch, 1.35 * inch, 1.35 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch, 1.2 * inch])
    tm.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), PAPER),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT_REG),
                ("FONTSIZE", (0, 0), (-1, -1), 7.6),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [SOFT_PINK, PAPER, PAPER, PAPER, PAPER, PAPER, PAPER]),
                ("FONTNAME", (0, 1), (-1, 1), FONT_BOLD),
                ("TEXTCOLOR", (0, 1), (2, 1), MAGENTA),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
                ("TOPPADDING", (0, 0), (-1, -1), 2.2),
            ]
        )
    )
    tm.wrapOn(c, 12.0 * inch, 1.5 * inch)
    tm.drawOn(c, 0.85 * inch, 0.70 * inch)
    c.showPage()

    # ================= SLIDE 5: INTERPRETABILITY & THRESHOLD STRATEGY =================
    draw_modern_content_bg(
        c,
        "Interpretability and Dual-Threshold Decision Policy",
        "Balancing Loss Prevention and Legitimate Customer Experience",
        5,
    )
    draw_card(c, 0.55 * inch, 2.72 * inch, 12.2 * inch, 3.58 * inch)
    fig3 = FIGURES_DIR / "fig3_feature_importance_threshold.png"
    if fig3.is_file():
        c.drawImage(str(fig3), 0.70 * inch, 2.80 * inch, width=11.9 * inch, height=3.42 * inch, preserveAspectRatio=True)

    draw_card(c, 0.55 * inch, 0.62 * inch, 12.2 * inch, 1.92 * inch, "Recommended Operating Policy by Score Band (Calibrated Probability p)", accent_bar=MAGENTA)
    policy_data = [
        ["Score Band (p)", "Risk Level", "Recommended Action", "Customer Experience", "Out-of-Time Performance (263,913 tx)"],
        ["p < 0.10", "LOW (90.6%)", "Instant Approval (Straight-Through)", "Zero friction (< 10 ms)", "No frauds outside the 4 critical regimes"],
        ["0.10 <= p < 0.288", "MODERATE (4.1%)", "Approval with Passive Monitoring", "No checkout block", "Keeps 97.8% of legitimate customers unbothered"],
        ["0.288 <= p < 0.500", "HIGH (2.3%)", "Step-Up Authentication (Push 2FA / App Biometrics)", "Quick 1-click confirmation", "Raises Recall from 53.2% to 73.1% (+2,133 frauds)"],
        ["p >= 0.500", "CRITICAL (3.0%)", "Automatic Preventive Block + Alert", "Immediate account protection", "70.8% Precision and only 0.93% FPR"],
    ]
    tp = Table(policy_data, colWidths=[1.5 * inch, 1.5 * inch, 3.1 * inch, 2.3 * inch, 3.6 * inch])
    tp.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), PAPER),
                ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
                ("FONTNAME", (0, 1), (-1, -1), FONT_REG),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [PAPER, PAPER, ICE_BLUE, SOFT_PINK]),
                ("TEXTCOLOR", (1, 4), (1, 4), MAGENTA),
                ("FONTNAME", (1, 4), (1, 4), FONT_BOLD),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    tp.wrapOn(c, 12.0 * inch, 1.4 * inch)
    tp.drawOn(c, 0.65 * inch, 0.74 * inch)
    c.showPage()

    # ================= SLIDE 6: JANUARY 2026 RESULTS =================
    draw_modern_content_bg(
        c,
        "Classification Results: January/2026 Data (44,451 Transactions)",
        "Risk Projection, Category Breakdown and Financial Impact",
        6,
    )
    draw_card(c, 0.55 * inch, 2.75 * inch, 12.2 * inch, 3.55 * inch)
    fig4 = FIGURES_DIR / "fig4_jan2026_predictions.png"
    if fig4.is_file():
        c.drawImage(str(fig4), 0.70 * inch, 2.83 * inch, width=11.9 * inch, height=3.38 * inch, preserveAspectRatio=True)

    draw_card(c, 0.55 * inch, 0.62 * inch, 12.2 * inch, 1.98 * inch, "Executive Summary of the Classified File (transacoes_jan2026_classificadas.csv)", accent_bar=ELECTRIC)
    p_jan = Paragraph(
        f"• <b>Seasonal Consistency:</b> The expected fraud volume in January/2026 (sum of probabilities) is <b>{jan['expected_fraud_volume_sum_probs']:.1f} transactions ({jan['expected_fraud_rate_mean_prob']*100:.2f}%)</b>, in line with the January/2025 rate (3.73%). This is an estimate: January has no labels.<br/>"
        f"• <b>Automatic Block (flag_fraude = True @ p >= 0.50):</b> <b>{jan['flagged_at_0_50']:,} transactions ({jan['flagged_rate_at_0_50']*100:.2f}%)</b> classified as high-certainty fraud, protecting <b>R$ {jan['amount_flagged_at_0_50_brl']:,.2f}</b> in direct transaction value.<br/>"
        f"• <b>Expanded Step-Up Coverage (flag_fraude_threshold_otimo = True @ p >= {jan['optimal_f1_threshold']:.3f}):</b> <b>{jan['flagged_at_optimal_f1']:,} transactions ({jan['flagged_rate_at_optimal_f1']*100:.2f}%)</b> under active protection (R$ {jan['amount_flagged_at_optimal_brl']:,.2f}), with {jan['risk_tier_counts'].get('BAIXO', 0):,} transactions ({jan['risk_tier_counts'].get('BAIXO', 0)/jan['total_transactions']*100:.1f}%) approved in the LOW tier with no friction.<br/>"
        f"• <b>Most Targeted Sectors in Jan/2026:</b> Fuel ({sector_rate('Fuel'):.2f}% predicted fraud rate @0.50) and Food & Beverage ({sector_rate('Food & Beverage'):.2f}%), followed by the other sectors concentrated in the 22h–23h window.",
        ParagraphStyle("BulletCompact", parent=bullet_style, fontSize=9.1, leading=12.8, spaceAfter=3),
    )
    p_jan.wrapOn(c, 11.7 * inch, 1.45 * inch)
    p_jan.drawOn(c, 0.75 * inch, 0.72 * inch)
    c.showPage()

    # ================= SLIDE 7: REAL-TIME KAFKA & DASHBOARD ARCHITECTURE =================
    draw_modern_content_bg(
        c,
        "Innovation: System Architecture (Kafka + Microservices + Colab)",
        "From Remote Modeling on Google Colab to Real-Time Operation",
        7,
    )
    draw_card(c, 0.55 * inch, 0.68 * inch, 12.2 * inch, 5.62 * inch)
    fig5 = FIGURES_DIR / "fig5_system_architecture.png"
    if fig5.is_file():
        c.drawImage(str(fig5), 0.72 * inch, 0.80 * inch, width=11.86 * inch, height=5.38 * inch, preserveAspectRatio=True)
    c.showPage()

    # ================= SLIDE 8: CODE & PROJECT ARCHITECTURE =================
    draw_modern_content_bg(
        c,
        "Code Architecture and Repository Layout",
        "Modules, Data Contracts, Automated Tests and Artifacts",
        8,
    )
    draw_card(c, 0.55 * inch, 0.68 * inch, 12.2 * inch, 5.62 * inch)
    fig6 = FIGURES_DIR / "fig6_code_project_structure.png"
    if fig6.is_file():
        c.drawImage(str(fig6), 0.72 * inch, 0.80 * inch, width=11.86 * inch, height=5.38 * inch, preserveAspectRatio=True)
    c.showPage()

    # ================= SLIDE 9: ROADMAP & RECOMMENDATIONS =================
    draw_modern_content_bg(
        c,
        "Strategic Recommendations and Next Steps",
        "Rollout Plan and Continuous Evolution of Fraud Prevention",
        9,
    )
    draw_card(c, 0.55 * inch, 0.75 * inch, 3.9 * inch, 5.45 * inch, "Curto Prazo (Semanas 1–4)", accent_bar=ELECTRIC)
    p_r1 = Paragraph(
        "<b>1. Retire the Legacy Score:</b><br/>Replace <i>score_interno</i> immediately with the probabilistic score of the LightGBM pipeline.<br/><br/>"
        "<b>2. Activate the Dual Threshold:</b><br/>Configure automatic block at <b>p >= 0.50</b> and a Step-Up challenge (2FA/biometrics) for <b>0.288 <= p < 0.50</b>.<br/><br/>"
        "<b>3. New-Device Alert Rule:</b><br/>Require device validation (Device Binding) on <i>mobile</i> purchases with <b>is_new_device=True</b> between 22h–23h and 12h–14h.",
        body_style,
    )
    p_r1.wrapOn(c, 3.45 * inch, 4.5 * inch)
    p_r1.drawOn(c, 0.75 * inch, 2.15 * inch)

    draw_card(c, 4.70 * inch, 0.75 * inch, 3.9 * inch, 5.45 * inch, "Medium Term (Months 2–3)", accent_bar=ROYAL)
    p_r2 = Paragraph(
        "<b>1. Real-Time Feature Store (Redis/Flink):</b><br/>Add sliding-window velocity counters (e.g., attempts per card/device in the last 5 and 15 minutes).<br/><br/>"
        "<b>2. Transactional Outbox Pattern:</b><br/>Evolve the PostgreSQL -> Kafka publication in the <i>Transaction Gateway</i> with an Outbox table for atomic <i>exactly-once</i> delivery.<br/><br/>"
        "<b>3. Chargeback Feedback Loop:</b><br/>Feed fraud confirmations and customer disputes directly into a Kafka topic for monthly retraining; with a second year of data, re-test seasonal (holiday) effects.",
        body_style,
    )
    p_r2.wrapOn(c, 3.45 * inch, 4.5 * inch)
    p_r2.drawOn(c, 4.90 * inch, 2.15 * inch)

    draw_card(c, 8.85 * inch, 0.75 * inch, 3.9 * inch, 5.45 * inch, "Governance and Deliverables in the .ZIP", accent_bar=MAGENTA)
    p_r3 = Paragraph(
        "<b>Delivered Content:</b><br/>"
        "• <b>Case_Fraude_Desenvolvimento.ipynb:</b> Single executable notebook with EDA, modeling and insights.<br/><br/>"
        "• <b>transacoes_jan2026_classificadas.csv:</b> Jan/2026 file (44,451 rows) with flag_fraude, probabilidade_fraude, flag_fraude_threshold_otimo e nivel_risco.<br/><br/>"
        "• <b>Apresentacao_Case_Fraude_Jan2026.pdf:</b> This executive presentation.<br/><br/>"
        "• <b>GitHub Repository (Kafka Architecture + Dashboard):</b><br/>"
        "<font color='#0046E1' size='8.6'><u>https://github.com/leorossi08/atf-challenge-fraud-detection</u></font>",
        body_style,
    )
    p_r3.wrapOn(c, 3.45 * inch, 4.5 * inch)
    p_r3.drawOn(c, 9.05 * inch, 1.85 * inch)
    c.showPage()

    c.save()
    print(f"Modern Artefact Presentation PDF generated at: {PDF_PATH}")


def build_zip():
    files_to_pack = [
        (DELIVERABLES_DIR / "transacoes_jan2026_classificadas.csv", "transacoes_jan2026_classificadas.csv"),
        (PDF_PATH, "Apresentacao_Case_Fraude_Jan2026.pdf"),
        (BASE_DIR / "Case_Fraude_Desenvolvimento.ipynb", "Case_Fraude_Desenvolvimento.ipynb"),
        (BASE_DIR / "models" / "fraud_models_cv.metrics.json", "extra_documentation/fraud_models_cv.metrics.json"),
    ]
    if (BASE_DIR / "README.md").is_file():
        files_to_pack.append((BASE_DIR / "README.md", "extra_documentation/README_Kafka_Dashboard_Architecture.md"))
    if (BASE_DIR / "ARCHITECTURE.md").is_file():
        files_to_pack.append((BASE_DIR / "ARCHITECTURE.md", "extra_documentation/ARCHITECTURE_Diagrams.md"))
    if (BASE_DIR / "model-evaluation.md").is_file():
        files_to_pack.append((BASE_DIR / "model-evaluation.md", "extra_documentation/model-evaluation.md"))
    if (BASE_DIR / "run.md").is_file():
        files_to_pack.append((BASE_DIR / "run.md", "extra_documentation/Demo_Run_Guide.md"))

    for fig in sorted(FIGURES_DIR.glob("*.png")):
        files_to_pack.append((fig, f"extra_documentation/figures/{fig.name}"))

    for target_zip in [ZIP_PATH, ZIP_DELIVERABLE_COPY]:
        with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for src, arcname in files_to_pack:
                if src.is_file():
                    zf.write(src, arcname=arcname)
        print(f"Created deliverable ZIP archive: {target_zip} ({target_zip.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    build_pdf()
    build_zip()
