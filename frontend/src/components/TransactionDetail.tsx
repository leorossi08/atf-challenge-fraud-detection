import { Activity, AlertTriangle, Check, MapPin } from "lucide-react";
import {
  formatDate,
  formatMoney,
  formatPercent,
  maskCardReference,
  type Transaction,
} from "../types/fraud";
import { CategoryIcon, DeviceIcon, StatusBadge } from "./Badges";

export function TransactionDetail({ transaction }: { transaction: Transaction }) {
  return (
    <div className="detail-content">
      <div className="detail-topline">
        <span className="section-kicker">RECORD INSPECTOR</span>
        <StatusBadge status={transaction.status} isFraud={transaction.isFraud} />
      </div>
      <h3 className="detail-id">{transaction.transactionId}</h3>
      <div className="detail-score-block">
        <span>Fraud probability</span>
        <strong className={transaction.isFraud ? "text-risk" : ""}>
          {formatPercent(transaction.fraudProbability)}
        </strong>
        {transaction.fraudProbability !== null && (
          <div className="large-score-track">
            <span style={{ width: `${Math.min(100, transaction.fraudProbability * 100)}%` }} />
          </div>
        )}
      </div>
      <dl className="detail-list">
        <div>
          <dt>Client / Card</dt>
          <dd>{maskCardReference(transaction.cardId)}</dd>
        </div>
        <div>
          <dt>Amount</dt>
          <dd>{formatMoney(transaction.amount)}</dd>
        </div>
        <div>
          <dt>Merchant</dt>
          <dd>
            <CategoryIcon category={transaction.tipo_estabelecimento} />{" "}
            {transaction.nome_estabelecimento ?? "N/A"}
          </dd>
        </div>
        <div>
          <dt>Location</dt>
          <dd>
            <MapPin size={11} /> {transaction.cidade ?? "N/A"} · {transaction.cidade_regiao ?? ""}
          </dd>
        </div>
        <div>
          <dt>Device</dt>
          <dd>
            <DeviceIcon deviceType={transaction.device_type} />{" "}
            {transaction.device_type ?? "mobile"}{" "}
            {transaction.is_new_device ? "· NEW" : "· Known"}
          </dd>
        </div>
        <div>
          <dt>Transaction time</dt>
          <dd>{formatDate(transaction.timestamp)}</dd>
        </div>
        <div>
          <dt>Received</dt>
          <dd>{formatDate(transaction.createdAt)}</dd>
        </div>
        <div>
          <dt>Last status update</dt>
          <dd>{formatDate(transaction.updatedAt)}</dd>
        </div>
      </dl>
      {transaction.message && (
        <div className="failure-message">
          <AlertTriangle size={14} />
          {transaction.message}
        </div>
      )}
      <div className="lifecycle">
        <span className="section-kicker">LIFECYCLE</span>
        <div className={`lifecycle-step${transaction.status !== "FAILED" ? " reached" : ""}`}>
          <span className="step-mark">
            <Check size={11} />
          </span>
          <span>Queued</span>
        </div>
        <div
          className={`lifecycle-step${
            transaction.status === "PROCESSING" || transaction.status === "COMPLETED"
              ? " reached"
              : ""
          }`}
        >
          <span className="step-mark">
            <Activity size={11} />
          </span>
          <span>{transaction.status === "PROCESSING" ? "Processing" : "Analysis"}</span>
        </div>
        <div
          className={`lifecycle-step${
            transaction.status === "COMPLETED" || transaction.status === "FAILED"
              ? " reached"
              : ""
          }`}
        >
          <span className="step-mark">
            <Check size={11} />
          </span>
          <span>{transaction.status === "FAILED" ? "Failed" : "Result stored"}</span>
        </div>
      </div>
    </div>
  );
}
