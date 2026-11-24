import { AlertTriangle, ShieldCheck } from "lucide-react";
import type { ModelMetrics } from "../types/fraud";

interface ModelQualityCardProps {
  metrics: ModelMetrics | null;
}

export function ModelQualityCard({ metrics }: ModelQualityCardProps) {
  return (
    <section className="model-section" id="model-quality">
      <div className="section-heading model-heading">
        <div>
          <div className="section-kicker">MODEL CARD · 8-MODEL BENCHMARK & JAN-2026 FORECAST</div>
          <h2>Model quality</h2>
        </div>
        <div className="model-chip">
          <span className="model-chip-dot" />{" "}
          {(metrics?.active_model ?? "Model").replaceAll("_", " ")} · active
        </div>
      </div>

      {metrics ? (
        <>
          <div className="model-summary">
            <div>
              <span>Validation strategy</span>
              <strong>Rolling-origin, {metrics.cross_validation.fold_count} months</strong>
            </div>
            <div>
              <span>Decision threshold</span>
              <strong>
                0.50 (Optimal F1:{" "}
                {metrics.january_2026_forecast
                  ? metrics.january_2026_forecast.optimal_f1_threshold.toFixed(2)
                  : "0.29"}
                )
              </strong>
            </div>
            <div>
              <span>Evaluation split</span>
              <strong>Out-of-time Jul–Dec/2025 (walk-forward)</strong>
            </div>
            <div>
              <span>Jan-2026 batch scored</span>
              <strong>
                {metrics.january_2026_forecast
                  ? `${metrics.january_2026_forecast.flagged_at_0_50.toLocaleString()} / ${metrics.january_2026_forecast.total_transactions.toLocaleString()} flagged`
                  : "44,451 transactions"}
              </strong>
            </div>
          </div>

          <div className="quality-table-wrap">
            <table className="quality-table">
              <thead>
                <tr>
                  <th>Model</th>
                  <th>Monthly PR-AUC</th>
                  <th>Pooled PR-AUC</th>
                  <th>Nov–Dec PR-AUC</th>
                  <th>ROC-AUC</th>
                  <th>Brier</th>
                  <th>F1 @0.50</th>
                  <th>F1 @Walk-fwd</th>
                  <th>Latency p50</th>
                  <th>Size</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.cross_validation.metrics)
                  .sort(([nameA], [nameB]) => {
                    const champ = metrics.champion_model ?? metrics.active_model;
                    if (nameA === champ) return -1;
                    if (nameB === champ) return 1;
                    const chronoA =
                      metrics.chronological_validation_nov_dec?.[nameA]?.average_precision ??
                      metrics.test_metrics_at_threshold_0_5[nameA]?.average_precision ??
                      0;
                    const chronoB =
                      metrics.chronological_validation_nov_dec?.[nameB]?.average_precision ??
                      metrics.test_metrics_at_threshold_0_5[nameB]?.average_precision ??
                      0;
                    return chronoB - chronoA;
                  })
                  .map(([name, values]) => {
                    const test = metrics.test_metrics_at_threshold_0_5[name];
                    const opt = metrics.test_metrics_at_optimal_f1_threshold?.[name];
                    const chrono = metrics.chronological_validation_nov_dec?.[name];
                    const ops = metrics.operational_tradeoffs?.[name];
                    const isChampion = name === (metrics.champion_model ?? metrics.active_model);

                    return (
                      <tr key={name} className={isChampion ? "champion-row" : ""}>
                        <td className="model-name">
                          {name.replaceAll("_", " ")}
                          {isChampion ? " ★" : ""}
                        </td>
                        <td>
                          {values.average_precision.mean.toFixed(4)}{" "}
                          <span className="std-value">
                            ± {values.average_precision.std.toFixed(4)}
                          </span>
                        </td>
                        <td>{test.average_precision.toFixed(4)}</td>
                        <td>{chrono ? chrono.average_precision.toFixed(4) : "--"}</td>
                        <td>{test.roc_auc !== undefined ? test.roc_auc.toFixed(4) : "--"}</td>
                        <td>{test.brier_score !== undefined ? test.brier_score.toFixed(5) : "--"}</td>
                        <td>{test.f1.toFixed(4)}</td>
                        <td>{opt ? opt.f1.toFixed(4) : "--"}</td>
                        <td>{ops ? `${ops.p50_ms.toFixed(1)} ms` : "--"}</td>
                        <td>
                          {ops
                            ? ops.artifact_size_kb >= 1024
                              ? `${(ops.artifact_size_kb / 1024).toFixed(1)} MB`
                              : `${ops.artifact_size_kb.toFixed(0)} KB`
                            : "--"}
                        </td>
                      </tr>
                    );
                  })}
              </tbody>
            </table>
          </div>

          <div className="model-footnote">
            <ShieldCheck size={15} /> Trained on Google Colab High-RAM VM over 523,525 historical
            2025 transactions and exported via joblib.
          </div>
        </>
      ) : (
        <div className="metrics-unavailable">
          <AlertTriangle size={17} />
          <span>
            Model metrics are unavailable. Check the inference service and its training report.
          </span>
        </div>
      )}
    </section>
  );
}
