import {
  ChevronLeft,
  ChevronRight,
  Fingerprint,
  RefreshCw,
  Search,
} from "lucide-react";
import {
  FILTERS,
  formatDate,
  formatMoney,
  formatPercent,
  maskCardReference,
  type Filter,
  type Transaction,
} from "../types/fraud";
import { CategoryIcon, DeviceIcon, StatusBadge } from "./Badges";
import { TransactionDetail } from "./TransactionDetail";

interface TransactionQueueProps {
  transactions: Transaction[];
  visibleTransactions: Transaction[];
  paginatedTransactions: Transaction[];
  selectedTransaction: Transaction | undefined;
  selectedId: string | null;
  filter: Filter;
  query: string;
  fraudCount: number;
  isRefreshing: boolean;
  page: number;
  totalPages: number;
  pageSize: number;
  pageStart: number;
  onFilterChange: (next: Filter) => void;
  onQueryChange: (next: string) => void;
  onSelectId: (id: string) => void;
  onRefresh: () => void;
  onPageChange: (next: number) => void;
  onPageSizeChange: (next: number) => void;
}

export function TransactionQueue({
  transactions,
  visibleTransactions,
  paginatedTransactions,
  selectedTransaction,
  selectedId,
  filter,
  query,
  fraudCount,
  isRefreshing,
  page,
  totalPages,
  pageSize,
  pageStart,
  onFilterChange,
  onQueryChange,
  onSelectId,
  onRefresh,
  onPageChange,
  onPageSizeChange,
}: TransactionQueueProps) {
  return (
    <section className="activity-section" id="data-status">
      <div className="section-heading">
        <div>
          <div className="section-kicker">LIVE WORK QUEUE</div>
          <h2>Recent transactions</h2>
        </div>
        <button
          className={`quiet-button${isRefreshing ? " refreshing" : ""}`}
          onClick={onRefresh}
          type="button"
          disabled={isRefreshing}
        >
          <RefreshCw size={15} /> Refresh
        </button>
      </div>

      <div className="table-toolbar">
        <div className="filter-tabs" role="tablist" aria-label="Filter transactions">
          {FILTERS.map((item) => (
            <button
              key={item.value}
              className={`filter-tab${filter === item.value ? " selected" : ""}`}
              onClick={() => onFilterChange(item.value)}
              role="tab"
              aria-selected={filter === item.value}
              type="button"
            >
              {item.label}
              {item.value === "FRAUD" && fraudCount > 0 && (
                <span className="tab-count">{fraudCount}</span>
              )}
            </button>
          ))}
        </div>
        <label className="search-field">
          <Search size={15} />
          <input
            value={query}
            onChange={(event) => onQueryChange(event.target.value)}
            placeholder="Search ID, client, city, device..."
            aria-label="Search by transaction or card ID"
          />
        </label>
      </div>

      <div className="table-layout">
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>Transaction</th>
                <th>Client / Card</th>
                <th>Merchant & City</th>
                <th>Device</th>
                <th>Amount</th>
                <th>Decision</th>
                <th>Fraud score</th>
                <th>Received</th>
              </tr>
            </thead>
            <tbody>
              {paginatedTransactions.map((transaction) => (
                <tr
                  key={transaction.transactionId}
                  className={selectedId === transaction.transactionId ? "row-selected" : ""}
                  onClick={() => onSelectId(transaction.transactionId)}
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      onSelectId(transaction.transactionId);
                    }
                  }}
                >
                  <td>
                    <span className="transaction-id">{transaction.transactionId}</span>
                  </td>
                  <td>
                    <span className="card-id">
                      <Fingerprint size={13} /> {maskCardReference(transaction.cardId)}
                    </span>
                  </td>
                  <td>
                    <span className="card-id">
                      <CategoryIcon category={transaction.tipo_estabelecimento} />{" "}
                      {transaction.nome_estabelecimento ?? "Merchant"} · {transaction.cidade ?? "BR"}
                    </span>
                  </td>
                  <td>
                    <span className={`device-pill${transaction.is_new_device ? " new-dev" : ""}`}>
                      <DeviceIcon deviceType={transaction.device_type} />
                      {transaction.device_type ?? "mobile"}
                      {transaction.is_new_device ? " · NEW" : ""}
                    </span>
                  </td>
                  <td className="amount-cell">{formatMoney(transaction.amount)}</td>
                  <td>
                    <StatusBadge status={transaction.status} isFraud={transaction.isFraud} />
                  </td>
                  <td>
                    {transaction.fraudProbability === null ? (
                      <span className="muted-value">--</span>
                    ) : (
                      <span
                        className={`score-value${transaction.isFraud ? " score-high" : ""}`}
                      >
                        {formatPercent(transaction.fraudProbability)}
                        <span className="score-track">
                          <span
                            style={{
                              width: `${Math.min(100, transaction.fraudProbability * 100)}%`,
                            }}
                          />
                        </span>
                      </span>
                    )}
                  </td>
                  <td className="time-cell">{formatDate(transaction.createdAt)}</td>
                </tr>
              ))}
              {!paginatedTransactions.length && (
                <tr>
                  <td className="empty-cell" colSpan={8}>
                    {transactions.length
                      ? "No transactions match these filters."
                      : "No transactions yet. Submit a demo transaction to start the flow."}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <aside className="detail-panel" aria-label="Selected transaction details">
          {selectedTransaction ? (
            <TransactionDetail transaction={selectedTransaction} />
          ) : (
            <div className="detail-empty">
              <div className="detail-icon">
                <Fingerprint size={18} />
              </div>
              <span className="section-kicker">RECORD INSPECTOR</span>
              <h3>Select a transaction</h3>
              <p>
                Its lifecycle, device channel, merchant category, score, and timestamps will appear
                here.
              </p>
            </div>
          )}
        </aside>
      </div>

      <div className="table-footer">
        <span>
          Showing{" "}
          {visibleTransactions.length
            ? `${pageStart + 1}–${Math.min(pageStart + pageSize, visibleTransactions.length)}`
            : "0"}{" "}
          of {visibleTransactions.length.toLocaleString()} matching (
          {transactions.length.toLocaleString()} total in view)
        </span>
        <div className="pagination-controls">
          <label className="page-size-select">
            <span>Rows</span>
            <select
              value={pageSize}
              onChange={(e) => onPageSizeChange(Number(e.target.value))}
              aria-label="Rows per page"
            >
              <option value={15}>15</option>
              <option value={25}>25</option>
              <option value={50}>50</option>
              <option value={100}>100</option>
            </select>
          </label>
          <button
            className="page-btn"
            type="button"
            onClick={() => onPageChange(1)}
            disabled={page <= 1}
            aria-label="First page"
          >
            «
          </button>
          <button
            className="page-btn"
            type="button"
            onClick={() => onPageChange(Math.max(1, page - 1))}
            disabled={page <= 1}
            aria-label="Previous page"
          >
            <ChevronLeft size={13} />
          </button>
          <span className="page-indicator">
            Page {page} of {totalPages}
          </span>
          <button
            className="page-btn"
            type="button"
            onClick={() => onPageChange(Math.min(totalPages, page + 1))}
            disabled={page >= totalPages}
            aria-label="Next page"
          >
            <ChevronRight size={13} />
          </button>
          <button
            className="page-btn"
            type="button"
            onClick={() => onPageChange(totalPages)}
            disabled={page >= totalPages}
            aria-label="Last page"
          >
            »
          </button>
        </div>
        <span className="polling-badge">
          <span className="live-dot" /> Polling Kafka gateway
        </span>
      </div>
    </section>
  );
}
