export type Status = "QUEUED" | "PROCESSING" | "COMPLETED" | "FAILED";
export type Filter = "ALL" | Status | "FRAUD";

export interface Transaction {
  transactionId: string;
  cardId: string;
  amount: number;
  timestamp: string;
  nome_estabelecimento?: string;
  tipo_estabelecimento?: string;
  cidade?: string;
  cidade_regiao?: string;
  device_type?: string;
  is_new_device?: boolean;
  turno?: string;
  status: Status;
  isFraud: boolean | null;
  fraudProbability: number | null;
  message: string | null;
  createdAt: string;
  updatedAt: string;
}

export interface MetricSummary {
  mean: number;
  std: number;
}

export interface ModelMetrics {
  active_model: string;
  champion_model?: string;
  cross_validation: {
    fold_count: number;
    metrics: Record<string, Record<string, MetricSummary>>;
  };
  test_metrics_at_threshold_0_5: Record<string, Record<string, number>>;
  test_metrics_at_optimal_f1_threshold?: Record<string, Record<string, number>>;
  chronological_validation_nov_dec?: Record<string, Record<string, number>>;
  operational_tradeoffs?: Record<
    string,
    { train_time_sec: number; artifact_size_kb: number; p50_ms: number; p95_ms: number }
  >;
  january_2026_forecast?: {
    total_transactions: number;
    flagged_at_0_50: number;
    flagged_rate_at_0_50: number;
    flagged_at_optimal_f1: number;
    optimal_f1_threshold: number;
  };
}

export const FILTERS: { label: string; value: Filter }[] = [
  { label: "All activity", value: "ALL" },
  { label: "Queued", value: "QUEUED" },
  { label: "Processing", value: "PROCESSING" },
  { label: "Completed", value: "COMPLETED" },
  { label: "Fraud alerts", value: "FRAUD" },
  { label: "Failed", value: "FAILED" },
];

export const ESTABLISHMENTS: { name: string; type: string }[] = [
  { name: "Gas Station", type: "Fuel" },
  { name: "Restaurant X", type: "Food & Beverage" },
  { name: "Restaurant Y", type: "Food & Beverage" },
  { name: "Online Shop 1", type: "E-commerce" },
  { name: "Online Shop 2", type: "E-commerce" },
  { name: "Electronics Store", type: "Retail" },
  { name: "Clothing Store", type: "Retail" },
  { name: "Supermarket A", type: "Supermarket" },
  { name: "Supermarket B", type: "Supermarket" },
  { name: "Pharmacy Z", type: "Pharmacy" },
];

export const CITIES: { city: string; region: string; temp: number }[] = [
  { city: "São Paulo", region: "Sudeste", temp: 22.0 },
  { city: "Rio de Janeiro", region: "Sudeste", temp: 28.0 },
  { city: "Belo Horizonte", region: "Sudeste", temp: 24.0 },
  { city: "Curitiba", region: "Sul", temp: 18.0 },
  { city: "Porto Alegre", region: "Sul", temp: 17.0 },
  { city: "Recife", region: "Nordeste", temp: 30.0 },
  { city: "Salvador", region: "Nordeste", temp: 29.0 },
];

export function formatDate(value: string) {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return new Intl.DateTimeFormat("en", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(parsed);
}

export function formatMoney(value: number) {
  return new Intl.NumberFormat("pt-BR", {
    style: "currency",
    currency: "BRL",
  }).format(value);
}

export function formatPercent(value: number | null) {
  return value === null ? "--" : `${(value * 100).toFixed(1)}%`;
}

export function maskCardReference(cardId: string) {
  if (cardId.startsWith("Cliente_")) return cardId;
  return cardId.length > 4 ? `••••${cardId.slice(-4)}` : "••••";
}

export function inferShiftAndCategory(hour: number, amount: number) {
  const turno =
    hour < 6 ? "madrugada" : hour < 12 ? "manha" : hour < 18 ? "tarde" : "noite";
  const valor_categoria =
    amount < 25.0 ? "baixo" : amount <= 80.0 ? "medio" : "alto";
  return { turno, valor_categoria };
}
