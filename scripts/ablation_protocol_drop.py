#!/usr/bin/env python3
"""
Attribution of the metric drop between the legacy protocol and the current out-of-time protocol.

Same models, same data; ONE factor changes per step so the drop can be split into measured pieces:

  S1  legacy features + random 80/20 + threshold tuned on test      (= the legacy numbers, reproduced)
  S2  new features    + random 80/20 + threshold tuned on test      -> effect of the FEATURE changes
  S3  S2, scored only on the test rows from Jul-Dec                 -> effect of the MONTH MIX (harder months/December)
  S4  new features + walk-forward (train on the past only), Jul-Dec, threshold tuned on pooled test
                                                                    -> effect of PAST-ONLY TRAINING (no leakage from
                                                                       future/same-month rows and repeated customers)
  S5  S4 but threshold chosen from prior months only (walk-forward) -> effect of the THRESHOLD protocol (= current)
  S4o legacy features + walk-forward, tuned on pooled test         -> feature effect under the proper split

Random-split steps are averaged over 3 seeds to expose split noise; walk-forward steps report the std over months.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "services" / "ml-inference-service"))
import train as T  # noqa: E402

MODELS = ["logistic_regression", "xgboost", "lightgbm"]
SEEDS = [2026, 7, 11]
N_JOBS = __import__("os").cpu_count() or 4
N_EST = 300
DATA = BASE_DIR / "data" / "transacoes_historico_2025.csv"
OUT_JSON = BASE_DIR / "deliverables" / "ablation_protocol_drop.json"
OUT_MD = BASE_DIR / "deliverables" / "ablation_protocol_drop.md"

LEGACY_NUMERIC = T.BASE_NUMERIC_FEATURES + ["month", "day"]
NEW_NUMERIC = T.ENGINEERED_NUMERIC_FEATURES


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def features(df, params, legacy: bool):
    """legacy=True -> old feature set (month/day numeric, hard-coded constants, no calendar features)."""
    f = T.engineer_features(df, params)
    if legacy:
        dt = pd.to_datetime(df["datahora_transacao"])
        f["month"] = dt.dt.month.to_numpy()
        f["day"] = dt.dt.day.to_numpy()
    return f


def model(name, legacy):
    return T.build_models(2026, N_EST, N_JOBS, LEGACY_NUMERIC if legacy else NEW_NUMERIC)[name]


def metrics(y, p, thr=None):
    """PR-AUC, ROC-AUC, F1@0.5 and F1 at a threshold (tuned on y, p itself if thr is None)."""
    if thr is None:
        thr, _ = T.find_optimal_f1_threshold(y, p)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "f1_at_0_5": float(f1_score(y, (p >= 0.5).astype(int), zero_division=0)),
        "f1_at_threshold": float(f1_score(y, (p >= thr).astype(int), zero_division=0)),
        "threshold": float(thr),
    }


def mean_dicts(ds):
    return {k: float(np.mean([d[k] for d in ds])) for k in ds[0]}


def main():
    df = pd.read_csv(DATA)
    y = df[T.TARGET_COLUMN].astype(int).to_numpy()
    month = pd.to_datetime(df["datahora_transacao"]).dt.month.to_numpy()
    idx = np.arange(len(df))
    results = {m: {} for m in MODELS}

    # ---------- random 80/20 steps (S1, S2, S3), averaged over seeds ----------
    for name in MODELS:
        s1, s2, s3 = [], [], []
        for seed in SEEDS:
            tr, te = train_test_split(idx, test_size=0.20, random_state=seed, stratify=y)
            # S1: legacy features, constants = legacy hard-coded defaults
            f_old = features(df, T.DEFAULT_DOMAIN_PARAMS, legacy=True)
            est = model(name, True)
            est.fit(f_old.iloc[tr], y[tr])
            s1.append(metrics(y[te], est.predict_proba(f_old.iloc[te])[:, 1]))
            # S2: new features, constants fit on the TRAIN rows only
            params = T.fit_domain_params(df.iloc[tr])
            f_new = features(df, params, legacy=False)
            est = model(name, False)
            est.fit(f_new.iloc[tr], y[tr])
            p = est.predict_proba(f_new.iloc[te])[:, 1]
            s2.append(metrics(y[te], p))
            # S3: same predictions, only test rows from Jul-Dec
            keep = month[te] >= 7
            s3.append(metrics(y[te][keep], p[keep]))
            log(f"{name} seed={seed}: S1 PR={s1[-1]['pr_auc']:.4f}  S2 PR={s2[-1]['pr_auc']:.4f}  S3 PR={s3[-1]['pr_auc']:.4f}")
        results[name]["S1"] = mean_dicts(s1)
        results[name]["S2"] = mean_dicts(s2)
        results[name]["S3"] = mean_dicts(s3)
        results[name]["S1_seed_pr_auc"] = [d["pr_auc"] for d in s1]

    # ---------- walk-forward steps (S4, S5, S4o) ----------
    scored = list(range(7, 13))
    for name in MODELS:
        probs_new, probs_old, ys = {}, {}, {}
        for m in range(6, 13):  # June is burn-in (only feeds the walk-forward threshold)
            tr, te = idx[month < m], idx[month == m]
            params = T.fit_domain_params(df.iloc[tr])
            f_new = features(df, params, legacy=False)
            est = model(name, False)
            est.fit(f_new.iloc[tr], y[tr])
            probs_new[m] = est.predict_proba(f_new.iloc[te])[:, 1]
            f_old = features(df, T.DEFAULT_DOMAIN_PARAMS, legacy=True)
            est = model(name, True)
            est.fit(f_old.iloc[tr], y[tr])
            probs_old[m] = est.predict_proba(f_old.iloc[te])[:, 1]
            ys[m] = y[te]
            log(f"{name} month={m}: new PR={average_precision_score(ys[m], probs_new[m]):.4f}  legacy-feat PR={average_precision_score(ys[m], probs_old[m]):.4f}")
        y_pool = np.concatenate([ys[m] for m in scored])
        p_new = np.concatenate([probs_new[m] for m in scored])
        p_old = np.concatenate([probs_old[m] for m in scored])
        results[name]["S4"] = metrics(y_pool, p_new)
        results[name]["S4o"] = metrics(y_pool, p_old)
        # S5: walk-forward threshold from prior months only
        thr_vec = []
        for m in scored:
            prior = [k for k in range(6, m)]
            thr, _ = T.find_optimal_f1_threshold(np.concatenate([ys[k] for k in prior]), np.concatenate([probs_new[k] for k in prior]))
            thr_vec.append(np.full(len(ys[m]), thr))
        thr_vec = np.concatenate(thr_vec)
        s5 = metrics(y_pool, p_new, thr=0.5)
        s5["f1_at_threshold"] = float(f1_score(y_pool, (p_new >= thr_vec).astype(int), zero_division=0))
        s5["threshold"] = float(np.mean(thr_vec))
        results[name]["S5"] = s5
        results[name]["S4_monthly_pr_auc"] = {str(m): float(average_precision_score(ys[m], probs_new[m])) for m in scored}

    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    labels = [
        ("S1", "Legacy: legacy features, random 80/20, threshold tuned on test"),
        ("S2", "+ new features (constants fit on train, calendar features, no raw month/day)"),
        ("S3", "+ score only the Jul–Dec rows of that random test set (month mix)"),
        ("S4", "+ train on the past only, walk-forward Jul–Dec, threshold still tuned on test"),
        ("S5", "+ walk-forward threshold from prior months only (= current protocol)"),
        ("S4o", "(side check) legacy features + walk-forward, threshold tuned on test"),
    ]
    lines = ["# Attribution of the metric drop (same models, one factor changed per step)", "",
             "Random-split rows are the mean of 3 seeds; walk-forward rows use all Jul–Dec 2025 out-of-time predictions.", ""]
    for name in MODELS:
        r = results[name]
        lines += [f"## {name}", "", "| Step | Condition | PR-AUC | Δ PR-AUC | ROC-AUC | F1 @0.50 | F1 @threshold | Δ F1 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        prev = None
        for key, label in labels:
            v = r[key]
            if key == "S4o":
                d_pr = d_f1 = ""
            else:
                d_pr = "" if prev is None else f"{v['pr_auc'] - prev['pr_auc']:+.4f}"
                d_f1 = "" if prev is None else f"{v['f1_at_threshold'] - prev['f1_at_threshold']:+.4f}"
                prev = v
            lines.append(f"| {key} | {label} | {v['pr_auc']:.4f} | {d_pr} | {v['roc_auc']:.4f} | {v['f1_at_0_5']:.4f} | {v['f1_at_threshold']:.4f} | {d_f1} |")
        total_pr = r["S5"]["pr_auc"] - r["S1"]["pr_auc"]
        total_f1 = r["S5"]["f1_at_threshold"] - r["S1"]["f1_at_threshold"]
        lines += ["", f"Total S1 → S5: PR-AUC {total_pr:+.4f}, F1 {total_f1:+.4f}. S1 seed spread (PR-AUC): {', '.join(f'{x:.4f}' for x in r['S1_seed_pr_auc'])}.", ""]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"saved {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
