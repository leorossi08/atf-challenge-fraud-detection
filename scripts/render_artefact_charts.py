#!/usr/bin/env python3
"""
Rerenders all 4 high-resolution (260 DPI) charts in Artefact's modern consulting visual identity
(Midnight Navy #0D1634, Electric Blue #0046E1, Artefact Magenta #FF0066, Cyan #00A8E8, clean
minimalist spines, Noto Sans Display/Noto Sans typography) using the pre-trained joblib models
and pre-scored Jan-2026 dataset (fast local execution, no model retraining).
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "services" / "ml-inference-service"))
from train import CATEGORICAL_FEATURES, ENGINEERED_NUMERIC_FEATURES

HIST_CSV = BASE_DIR / "data" / "transacoes_historico_2025.csv"
SCORED_JAN_CSV = BASE_DIR / "deliverables" / "transacoes_jan2026_classificadas.csv"
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "fraud_models_cv.metrics.json"
FIGURES_DIR = BASE_DIR / "deliverables" / "figures"

# Artefact Palette
NAVY = "#0D1634"
ROYAL = "#001994"
ELECTRIC = "#0046E1"
CYAN = "#1890D7"
MAGENTA = "#FF0066"
SOFT_PINK = "#FF4D94"
SLATE = "#64748B"
GRID_COL = "#E6ECF5"
BG_WHITE = "#FFFFFF"


def configure_style():
    for fp in [
        "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansDisplay-Bold.ttf",
    ]:
        if Path(fp).is_file():
            font_manager.fontManager.addfont(fp)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans", "DejaVu Sans"],
            "axes.facecolor": BG_WHITE,
            "figure.facecolor": BG_WHITE,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.labelcolor": NAVY,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": NAVY,
            "grid.color": GRID_COL,
            "grid.linestyle": "-",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def main():
    configure_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    champ = report["champion_model"]
    opt_t = report["test_metrics_at_optimal_f1_threshold"][champ]["threshold"]

    print("[1/4] Rendering Fig 1 (Artefact EDA Heatmap + Device Lift)...")
    df_hist = pd.read_csv(HIST_CSV)
    dt = pd.to_datetime(df_hist["datahora_transacao"])
    df_hist["hour"] = dt.dt.hour

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9), dpi=240)

    # Custom Artefact Navy -> Electric Blue -> Magenta colormap
    art_cmap = LinearSegmentedColormap.from_list(
        "artefact_heat", ["#F4F7FC", "#BDD4FF", "#0046E1", "#7B1FA2", "#FF0066"]
    )
    pivot = (
        df_hist.pivot_table(
            index="tipo_estabelecimento",
            columns="hour",
            values="flag_fraude",
            aggfunc="mean",
        )
        * 100.0
    )
    im = axes[0].imshow(pivot.values, aspect="auto", cmap=art_cmap)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index, fontsize=9.5, fontweight="bold")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xticklabels([f"{h:02d}h" for h in range(0, 24, 2)], fontsize=8.5)
    axes[0].set_title("Fraud Rate (%) by Merchant Category and Hour of Day", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[0].set_xlabel("Transaction Hour (2025)", fontsize=9.5, labelpad=6)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.042, pad=0.03)
    cbar.outline.set_visible(False)
    cbar.set_label("% Fraud", fontsize=9)

    dev_pivot = (
        df_hist.groupby(["device_type", "is_new_device"])["flag_fraude"].mean().unstack()
        * 100.0
    )
    x = np.arange(len(dev_pivot.index))
    width = 0.34
    b1 = axes[1].bar(x - width / 2, dev_pivot[False], width, label="Known Device", color=NAVY, zorder=3)
    b2 = axes[1].bar(x + width / 2, dev_pivot[True], width, label="New Device (is_new_device=True)", color=MAGENTA, zorder=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([d.capitalize() for d in dev_pivot.index], fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Fraud Rate (%)", fontsize=9.5)
    axes[1].set_ylim(0, max(dev_pivot[True]) * 1.24)
    axes[1].set_title("Amplifying Factor: New Device by Channel", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[1].legend(frameon=False, fontsize=9, loc="upper right")
    axes[1].grid(axis="y", zorder=0)
    for bars, col in [(b1, NAVY), (b2, MAGENTA)]:
        for bar in bars:
            h = bar.get_height()
            axes[1].annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color=col,
            )
    fig.tight_layout(w_pad=3.0)
    fig.savefig(FIGURES_DIR / "fig1_eda_regimes_device.png", bbox_inches="tight", facecolor=BG_WHITE)
    plt.close(fig)

    print("[2/4] Loading pooled out-of-time (rolling-origin Jul-Dec/2025) predictions for Fig 2 & Fig 3...")
    oot = np.load(BASE_DIR / "deliverables" / "oot_predictions.npz")
    test_y = oot["y"].astype(np.int32)
    n_test = len(test_y)
    names = ["lightgbm", "xgboost", "catboost", "soft_voting_ensemble", "logistic_regression"]
    model_probs = {name: oot[name].astype(float) for name in names}
    models = {"xgboost": joblib.load(MODELS_DIR / "xgboost_fraud_model_cv.joblib")}  # final fit, used for feature importance only

    # Fig 2: ROC & PR Curves (Top Models from 8-Model Benchmark)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9), dpi=240)
    styles = {
        "lightgbm": (MAGENTA, 2.5, "LightGBM" + (" ★ (Champion)" if champ == "lightgbm" else "")),
        "soft_voting_ensemble": (ELECTRIC, 2.0, "Soft-Voting Ensemble (3 GBDTs)"),
        "xgboost": (CYAN, 1.8, "XGBoost (Regularized)"),
        "catboost": ("#7C3AED", 1.7, "CatBoost (Ordered Boosting)"),
        "logistic_regression": (NAVY, 1.4, "Logistic Regression (Baseline)"),
    }
    for name in ["lightgbm", "soft_voting_ensemble", "xgboost", "catboost", "logistic_regression"]:
        col, lw, label = styles[name]
        probs = model_probs[name]
        fpr, tpr, _ = roc_curve(test_y, probs)
        roc_auc = roc_auc_score(test_y, probs)
        axes[0].plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.4f})", color=col, linewidth=lw)

        prec, rec, _ = precision_recall_curve(test_y, probs)
        pr_auc = average_precision_score(test_y, probs)
        axes[1].plot(rec, prec, label=f"{label} (PR={pr_auc:.4f})", color=col, linewidth=lw)

    axes[0].fill_between(*roc_curve(test_y, model_probs["lightgbm"])[:2], color=ELECTRIC, alpha=0.06)
    axes[0].plot([0, 1], [0, 1], "--", color="#94A3B8", linewidth=1)
    axes[0].set_title(f"ROC Curve — Rolling-Origin Jul–Dec/2025 ({n_test:,} out-of-time tx)", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[0].set_xlabel("False Positive Rate (FPR)", fontsize=9.5)
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=9.5)
    axes[0].legend(loc="lower right", frameon=True, facecolor=BG_WHITE, edgecolor=GRID_COL, fontsize=9)
    axes[0].grid(True)

    baseline_rate = float(np.mean(test_y))
    axes[1].axhline(baseline_rate, linestyle="--", color="#94A3B8", linewidth=1.2, label=f"Random Baseline ({baseline_rate*100:.1f}%)")
    axes[1].set_title("Precision-Recall Curve — Rolling-Origin Jul–Dec/2025", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=9.5)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=9.5)
    axes[1].legend(loc="lower left", frameon=True, facecolor=BG_WHITE, edgecolor=GRID_COL, fontsize=9)
    axes[1].grid(True)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(FIGURES_DIR / "fig2_roc_pr_curves.png", bbox_inches="tight", facecolor=BG_WHITE)
    plt.close(fig)

    print("[3/4] Rendering Fig 3 (Feature Importance + Threshold Trade-off)...")
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9), dpi=240)
    pre = models["xgboost"].named_steps["preprocessor"]
    clf = models["xgboost"].named_steps["classifier"]
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES))
    all_feat_names = ENGINEERED_NUMERIC_FEATURES + cat_names
    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-10:]
    bar_colors = [MAGENTA if i >= len(top_idx) - 2 else ELECTRIC if i >= len(top_idx) - 5 else NAVY for i in range(len(top_idx))]
    bars = axes[0].barh([all_feat_names[i] for i in top_idx], importances[top_idx], color=bar_colors, height=0.65, zorder=3)
    axes[0].set_title("Top 10 Most Important Features (XGBoost Gain)", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[0].set_xlabel("Relative Importance", fontsize=9.5)
    axes[0].set_xlim(0, max(importances[top_idx]) * 1.18)
    axes[0].grid(axis="x", zorder=0)
    for bar in bars:
        w = bar.get_width()
        axes[0].annotate(f"{w*100:.1f}%", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(4, 0),
                         textcoords="offset points", va="center", fontsize=8.5, fontweight="bold", color=NAVY)

    xgb_probs = model_probs[champ] if champ in model_probs else model_probs["xgboost"]
    ths = np.linspace(0.05, 0.90, 86)
    precs, recs, f1s = [], [], []
    for t in ths:
        p = (xgb_probs >= t).astype(int)
        precs.append(precision_score(test_y, p, zero_division=0))
        recs.append(recall_score(test_y, p, zero_division=0))
        f1s.append(f1_score(test_y, p, zero_division=0))
    axes[1].axvspan(opt_t, 0.50, color=ELECTRIC, alpha=0.08, label="Step-Up Auth Band (2FA)")
    axes[1].axvspan(0.50, 0.90, color=MAGENTA, alpha=0.06, label="Automatic Block Band")
    axes[1].plot(ths, precs, label="Precision", color=ELECTRIC, linewidth=2.2)
    axes[1].plot(ths, recs, label="Recall", color=MAGENTA, linewidth=2.2)
    axes[1].plot(ths, f1s, label="F1-Score", color=NAVY, linewidth=2.5)
    axes[1].axvline(0.50, linestyle="--", color=MAGENTA, linewidth=1.5)
    axes[1].axvline(opt_t, linestyle=":", color=ELECTRIC, linewidth=1.8)
    axes[1].set_title("Decision Curve and Dual-Threshold Policy", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[1].set_xlabel("Probability Threshold (p)", fontsize=9.5)
    axes[1].set_ylabel("Metric", fontsize=9.5)
    axes[1].legend(frameon=True, facecolor=BG_WHITE, edgecolor=GRID_COL, fontsize=8.2, loc="lower left")
    axes[1].grid(True)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(FIGURES_DIR / "fig3_feature_importance_threshold.png", bbox_inches="tight", facecolor=BG_WHITE)
    plt.close(fig)

    print("[4/4] Rendering Fig 4 (Jan-2026 Forecast Distribution & Category Breakdown)...")
    df_jan_scored = pd.read_csv(SCORED_JAN_CSV)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.9), dpi=240)
    axes[0].hist(df_jan_scored["probabilidade_fraude"], bins=40, color=NAVY, edgecolor=BG_WHITE, linewidth=0.5, log=True, zorder=3)
    axes[0].axvline(0.50, color=MAGENTA, linestyle="--", linewidth=2.0, label="Automatic Block (p = 0.50)")
    axes[0].axvline(opt_t, color=ELECTRIC, linestyle=":", linewidth=2.2, label=f"Step-Up Auth 2FA (p = {opt_t:.2f})")
    axes[0].set_title("Probability Distribution — January/2026 (Log Scale)", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[0].set_xlabel("Estimated Fraud Probability", fontsize=9.5)
    axes[0].set_ylabel("Transaction Volume (log)", fontsize=9.5)
    axes[0].legend(frameon=True, facecolor=BG_WHITE, edgecolor=GRID_COL, fontsize=8.8)
    axes[0].grid(True, zorder=0)

    jan_by_tipo = (
        df_jan_scored.groupby("tipo_estabelecimento")["flag_fraude"]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=True)
    )
    cat_cols = [NAVY, NAVY, ELECTRIC, ELECTRIC, MAGENTA, MAGENTA]
    axes[1].barh(jan_by_tipo.index, jan_by_tipo["mean"] * 100.0, color=cat_cols, height=0.62, zorder=3)
    for idx, (_, row) in enumerate(jan_by_tipo.iterrows()):
        axes[1].annotate(
            f"  {row['mean']*100:.2f}% ({int(row['sum'])} frauds)",
            xy=(row["mean"] * 100.0, idx),
            va="center",
            fontsize=9,
            fontweight="bold",
            color=MAGENTA if idx >= 4 else NAVY,
        )
    axes[1].set_xlim(0, max(jan_by_tipo["mean"] * 100.0) * 1.42)
    axes[1].set_title("Predicted Fraud Rate in January/2026 by Category", fontsize=11.5, fontweight="bold", pad=12, loc="left")
    axes[1].set_xlabel("Predicted Fraud Rate @0.50 (%)", fontsize=9.5)
    axes[1].grid(axis="x", zorder=0)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(FIGURES_DIR / "fig4_jan2026_predictions.png", bbox_inches="tight", facecolor=BG_WHITE)
    plt.close(fig)
    print("All 4 Artefact-branded figures rendered successfully.")


if __name__ == "__main__":
    main()
