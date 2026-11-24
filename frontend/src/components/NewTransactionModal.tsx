import { useState, type FormEvent } from "react";
import { Activity, CircleHelp, RefreshCw, X } from "lucide-react";
import { CITIES, ESTABLISHMENTS, inferShiftAndCategory } from "../types/fraud";

interface NewTransactionModalProps {
  isOpen: boolean;
  isSubmitting: boolean;
  onClose: () => void;
  onSubmitPayload: (payload: Record<string, unknown>) => Promise<void>;
  onValidationError: (message: string) => void;
}

export function NewTransactionModal({
  isOpen,
  isSubmitting,
  onClose,
  onSubmitPayload,
  onValidationError,
}: NewTransactionModalProps) {
  const [transactionId, setTransactionId] = useState("");
  const [cardId, setCardId] = useState("");
  const [amount, setAmount] = useState("28.50");
  const [merchantName, setMerchantName] = useState("Gas Station");
  const [cityName, setCityName] = useState("São Paulo");
  const [deviceType, setDeviceType] = useState("mobile");
  const [isNewDevice, setIsNewDevice] = useState(true);
  const [txHour, setTxHour] = useState("23");

  if (!isOpen) return null;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const numericAmount = Number(amount);
    if (!Number.isFinite(numericAmount) || numericAmount < 0) {
      onValidationError("Enter a valid non-negative amount.");
      return;
    }

    const est = ESTABLISHMENTS.find((e) => e.name === merchantName) ?? ESTABLISHMENTS[0];
    const cityObj = CITIES.find((c) => c.city === cityName) ?? CITIES[0];
    const hourNum = Math.max(0, Math.min(23, Number(txHour) || 22));
    const { turno, valor_categoria } = inferShiftAndCategory(hourNum, numericAmount);
    const timestampStr = `2026-01-15 ${String(hourNum).padStart(2, "0")}:30:00`;

    const payload = {
      transactionId: transactionId.trim() || `jan26-${crypto.randomUUID().slice(0, 8)}`,
      cardId: cardId.trim() || "Cliente_16241",
      amount: numericAmount,
      timestamp: timestampStr,
      nome_estabelecimento: est.name,
      tipo_estabelecimento: est.type,
      cidade: cityObj.city,
      cidade_regiao: cityObj.region,
      device_type: deviceType,
      is_new_device: isNewDevice,
      temperatura_celsius: cityObj.temp,
      codigo_postal: "01310-100",
      score_interno: 520,
      turno,
      valor_categoria,
    };

    await onSubmitPayload(payload);
    setTransactionId("");
    setCardId("");
  }

  return (
    <div
      className="dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="transaction-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="dialog-title"
      >
        <div className="dialog-heading">
          <div>
            <div className="section-kicker">NEW ANALYSIS</div>
            <h2 id="dialog-title">Submit a demo transaction</h2>
          </div>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close dialog"
            type="button"
          >
            <X size={18} />
          </button>
        </div>

        <p className="dialog-copy">
          Simulate a January-2026 transaction across merchant categories, cities, device types, and
          hours.
        </p>

        <form className="dialog-form" onSubmit={(event) => void handleSubmit(event)}>
          <label className="form-label">
            Transaction ID <span>Optional</span>
            <input
              value={transactionId}
              onChange={(event) => setTransactionId(event.target.value)}
              placeholder="Generated if blank"
              maxLength={100}
            />
          </label>

          <label className="form-label">
            Client / Card reference
            <input
              value={cardId}
              onChange={(event) => setCardId(event.target.value)}
              placeholder="e.g. card-demo-01"
              required
              maxLength={100}
            />
          </label>

          <label className="form-label">
            Amount (BRL)
            <div className="currency-input">
              <span>R$</span>
              <input
                type="number"
                value={amount}
                onChange={(event) => setAmount(event.target.value)}
                min="0"
                step="0.01"
                required
              />
            </div>
          </label>

          <label className="form-label">
            Merchant
            <select value={merchantName} onChange={(e) => setMerchantName(e.target.value)}>
              {ESTABLISHMENTS.map((e) => (
                <option key={e.name} value={e.name}>
                  {e.name} ({e.type})
                </option>
              ))}
            </select>
          </label>

          <label className="form-label">
            City & Region
            <select value={cityName} onChange={(e) => setCityName(e.target.value)}>
              {CITIES.map((c) => (
                <option key={c.city} value={c.city}>
                  {c.city} ({c.region})
                </option>
              ))}
            </select>
          </label>

          <div className="form-grid-2">
            <label className="form-label">
              Device Type
              <select value={deviceType} onChange={(e) => setDeviceType(e.target.value)}>
                <option value="mobile">mobile</option>
                <option value="desktop">desktop</option>
                <option value="tablet">tablet</option>
              </select>
            </label>
            <label className="form-label">
              Hour of Day (0-23)
              <input
                type="number"
                min="0"
                max="23"
                value={txHour}
                onChange={(e) => setTxHour(e.target.value)}
              />
            </label>
          </div>

          <label className="checkbox-field">
            <input
              type="checkbox"
              checked={isNewDevice}
              onChange={(e) => setIsNewDevice(e.target.checked)}
            />
            <span>
              Unrecognized / New Device (<code>is_new_device = true</code>)
            </span>
          </label>

          <div className="synthetic-note">
            <CircleHelp size={15} />
            <span>
              High-risk windows: 22h–23h (all merchants), 12h–14h (Food & Beverage in Sul/Sudeste),
              and 00h–08h (Fuel in SP/RJ), amplified by new mobile devices.
            </span>
          </div>

          <div className="dialog-actions">
            <button className="quiet-button" onClick={onClose} type="button">
              Cancel
            </button>
            <button className="primary-button" disabled={isSubmitting} type="submit">
              {isSubmitting ? <RefreshCw className="spin" size={16} /> : <Activity size={16} />}
              {isSubmitting ? "Submitting" : "Submit for analysis"}
            </button>
          </div>
        </form>
      </section>
    </div>
  );
}
