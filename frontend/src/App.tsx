import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ChevronRight,
  Clock3,
  Database,
  Gauge,
  Plus,
  ShieldCheck,
  X,
} from "lucide-react";
import { KpiSummary } from "./components/KpiSummary";
import { ModelQualityCard } from "./components/ModelQualityCard";
import { NewTransactionModal } from "./components/NewTransactionModal";
import { TransactionQueue } from "./components/TransactionQueue";
import type { Filter, ModelMetrics, Transaction } from "./types/fraud";

function App() {
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [filter, setFilter] = useState<Filter>("ALL");
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refreshTransactions() {
    setIsRefreshing(true);
    try {
      const response = await fetch("/api/transactions?limit=10000");
      if (!response.ok) throw new Error(`Gateway returned ${response.status}`);
      const data = (await response.json()) as Transaction[];
      setTransactions(data);
      setError(null);
    } catch {
      setError("Could not reach the gateway. Start the services and refresh.");
    } finally {
      setIsRefreshing(false);
    }
  }

  async function refreshMetrics() {
    try {
      const response = await fetch("/api/model/metrics");
      if (!response.ok) throw new Error(`Inference service returned ${response.status}`);
      setMetrics((await response.json()) as ModelMetrics);
    } catch {
      setMetrics(null);
    }
  }

  useEffect(() => {
    void refreshTransactions();
    void refreshMetrics();
    const intervalId = window.setInterval(() => {
      void refreshTransactions();
    }, 5000);
    return () => window.clearInterval(intervalId);
  }, []);

  useEffect(() => {
    setPage(1);
  }, [filter, query, pageSize]);

  const visibleTransactions = transactions.filter((transaction) => {
    const matchesFilter =
      filter === "ALL" ||
      (filter === "FRAUD" ? transaction.isFraud === true : transaction.status === filter);
    const normalizedQuery = query.trim().toLowerCase();
    const matchesQuery =
      !normalizedQuery ||
      transaction.transactionId.toLowerCase().includes(normalizedQuery) ||
      transaction.cardId.toLowerCase().includes(normalizedQuery) ||
      (transaction.nome_estabelecimento ?? "").toLowerCase().includes(normalizedQuery) ||
      (transaction.cidade ?? "").toLowerCase().includes(normalizedQuery) ||
      (transaction.device_type ?? "").toLowerCase().includes(normalizedQuery);
    return matchesFilter && matchesQuery;
  });

  const totalPages = Math.max(1, Math.ceil(visibleTransactions.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const pageStart = (safePage - 1) * pageSize;
  const paginatedTransactions = visibleTransactions.slice(pageStart, pageStart + pageSize);

  const selectedTransaction = transactions.find(
    (transaction) => transaction.transactionId === selectedId,
  );
  const fraudCount = transactions.filter((transaction) => transaction.isFraud === true).length;
  const activeCount = transactions.filter(
    (transaction) => transaction.status === "QUEUED" || transaction.status === "PROCESSING",
  ).length;
  const completedTransactions = transactions.filter(
    (transaction) => transaction.status === "COMPLETED",
  );
  const averageRisk = completedTransactions.length
    ? completedTransactions.reduce(
        (total, transaction) => total + (transaction.fraudProbability ?? 0),
        0,
      ) / completedTransactions.length
    : null;

  async function handleSubmitTransaction(payload: Record<string, unknown>) {
    setIsSubmitting(true);
    setError(null);
    try {
      const response = await fetch("/api/transactions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const body = (await response.json().catch(() => null)) as {
          detail?: string;
          error?: string;
        } | null;
        throw new Error(body?.detail ?? body?.error ?? `Request failed (${response.status})`);
      }
      const result = (await response.json()) as { transactionId: string };
      setSelectedId(result.transactionId);
      setIsDialogOpen(false);
      await refreshTransactions();
    } catch (submitError) {
      setError(
        submitError instanceof Error ? submitError.message : "Could not submit transaction.",
      );
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <div className="app-shell">
      <aside className="side-rail" aria-label="Primary navigation">
        <div className="brand-mark">
          <ShieldCheck size={21} strokeWidth={2.2} />
        </div>
        <nav className="rail-nav">
          <a className="rail-link active" href="#overview" aria-label="Overview" title="Overview">
            <Activity size={19} />
          </a>
          <a className="rail-link" href="#data-status" aria-label="Live Queue" title="Live Queue">
            <Database size={19} />
          </a>
          <a
            className="rail-link"
            href="#model-quality"
            aria-label="Model quality"
            title="Model quality"
          >
            <Gauge size={19} />
          </a>
        </nav>
        <div className="rail-bottom">
          <div className="avatar" aria-label="Artefact Data & AI Case">
            ATF
          </div>
        </div>
      </aside>

      <main className="workspace" id="overview">
        <header className="topbar">
          <div className="breadcrumb">
            <span>ARTEFACT · Risk Systems</span>
            <ChevronRight size={14} />
            <strong>Fraud Operations · Case Jan-2026</strong>
          </div>
          <div className="topbar-right">
            <span className="environment">
              <span className="environment-dot" /> KAFKA + ML STREAMING
            </span>
            <span className="topbar-divider" />
            <span className="updated-label">
              <Clock3 size={14} /> Refreshes every 5s
            </span>
          </div>
        </header>

        <div className="page-content">
          <section className="page-heading">
            <div>
              <div className="eyebrow">
                <span className="eyebrow-rule" /> ARTEFACT DATA & AI · REAL-TIME MONITORING
              </div>
              <h1>Fraud operations</h1>
              <p className="page-subtitle">
                Event-driven Kafka classification and ML benchmark across Brazilian merchant
                transactions (2025–Jan/2026).
              </p>
            </div>
            <button
              className="primary-button"
              onClick={() => setIsDialogOpen(true)}
              type="button"
            >
              <Plus size={17} /> New demo transaction
            </button>
          </section>

          {error && (
            <div className="notice error-notice" role="alert">
              <AlertTriangle size={17} /> <span>{error}</span>
              <button
                className="icon-button notice-close"
                onClick={() => setError(null)}
                aria-label="Dismiss error"
                type="button"
              >
                <X size={16} />
              </button>
            </div>
          )}

          <KpiSummary
            totalCount={transactions.length}
            activeCount={activeCount}
            fraudCount={fraudCount}
            averageRisk={averageRisk}
          />

          <TransactionQueue
            transactions={transactions}
            visibleTransactions={visibleTransactions}
            paginatedTransactions={paginatedTransactions}
            selectedTransaction={selectedTransaction}
            selectedId={selectedId}
            filter={filter}
            query={query}
            fraudCount={fraudCount}
            isRefreshing={isRefreshing}
            page={safePage}
            totalPages={totalPages}
            pageSize={pageSize}
            pageStart={pageStart}
            onFilterChange={setFilter}
            onQueryChange={setQuery}
            onSelectId={setSelectedId}
            onRefresh={() => void refreshTransactions()}
            onPageChange={setPage}
            onPageSizeChange={setPageSize}
          />

          <ModelQualityCard metrics={metrics} />

          <footer className="page-footer">
            <span>ARTEFACT · Fraud Detection Case · Event-Driven Architecture</span>
            <span>Local portfolio environment · Kafka remains private</span>
          </footer>
        </div>
      </main>

      <NewTransactionModal
        isOpen={isDialogOpen}
        isSubmitting={isSubmitting}
        onClose={() => setIsDialogOpen(false)}
        onSubmitPayload={handleSubmitTransaction}
        onValidationError={setError}
      />
    </div>
  );
}

export default App;
