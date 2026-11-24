import {
  AlertTriangle,
  Check,
  Clock3,
  Fuel,
  Globe,
  Monitor,
  Pill,
  ShoppingBag,
  ShoppingCart,
  Smartphone,
  Store,
  Tablet,
  Utensils,
  X,
} from "lucide-react";
import type { Status } from "../types/fraud";

export function DeviceIcon({ deviceType }: { deviceType?: string }) {
  const norm = (deviceType ?? "mobile").toLowerCase();
  if (norm === "desktop") return <Monitor size={13} aria-label="Desktop" />;
  if (norm === "tablet") return <Tablet size={13} aria-label="Tablet" />;
  return <Smartphone size={13} aria-label="Mobile" />;
}

export function CategoryIcon({ category }: { category?: string }) {
  switch (category) {
    case "Fuel":
      return <Fuel size={13} />;
    case "Food & Beverage":
      return <Utensils size={13} />;
    case "E-commerce":
      return <Globe size={13} />;
    case "Pharmacy":
      return <Pill size={13} />;
    case "Supermarket":
      return <ShoppingCart size={13} />;
    case "Retail":
      return <ShoppingBag size={13} />;
    default:
      return <Store size={13} />;
  }
}

export function StatusBadge({ status, isFraud }: { status: Status; isFraud: boolean | null }) {
  if (status === "COMPLETED" && isFraud) {
    return (
      <span className="status-badge fraud">
        <AlertTriangle size={13} /> Fraud flagged
      </span>
    );
  }
  const icon =
    status === "COMPLETED" ? (
      <Check size={13} />
    ) : status === "FAILED" ? (
      <X size={13} />
    ) : (
      <Clock3 size={13} />
    );
  return (
    <span className={`status-badge ${status.toLowerCase()}`}>
      {icon}
      {status.toLowerCase()}
    </span>
  );
}
