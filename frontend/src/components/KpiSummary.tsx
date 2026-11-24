import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  Clock3,
  Database,
  Gauge,
  ShieldAlert,
} from "lucide-react";
import { formatPercent } from "../types/fraud";

interface KpiSummaryProps {
  totalCount: number;
  activeCount: number;
  fraudCount: number;
  averageRisk: number | null;
}

export function KpiSummary({
  totalCount,
  activeCount,
  fraudCount,
  averageRisk,
}: KpiSummaryProps) {
  return (
    <section className="metrics-grid" aria-label="Transaction summary">
      <article className="metric-block">
        <div className="metric-top">
          <span>Transactions in view</span>
          <Activity size={17} />
        </div>
        <strong>{totalCount.toLocaleString()}</strong>
        <small>
          <span className="metric-neutral">
            <Database size={12} /> Up to 10,000 Kafka events
          </span>
        </small>
      </article>
      <article className="metric-block">
        <div className="metric-top">
          <span>Awaiting analysis</span>
          <Clock3 size={17} />
        </div>
        <strong>{activeCount}</strong>
        <small>
          <span className="metric-neutral">Queued or processing</span>
        </small>
      </article>
      <article className="metric-block risk-metric">
        <div className="metric-top">
          <span>Fraud flagged</span>
          <ShieldAlert size={17} />
        </div>
        <strong>{fraudCount}</strong>
        <small>
          <span className="metric-risk">
            <ArrowUpRight size={13} /> Confirmed model flags
          </span>
        </small>
      </article>
      <article className="metric-block">
        <div className="metric-top">
          <span>Average fraud score</span>
          <Gauge size={17} />
        </div>
        <strong>{formatPercent(averageRisk)}</strong>
        <small>
          <span className="metric-neutral">
            <ArrowDownRight size={13} /> Completed records in view
          </span>
        </small>
      </article>
    </section>
  );
}
