import { useMemo, useState } from "react";
import Papa from "papaparse";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

const FRAUD_COLOR = "#d73a31";
const SAFE_COLOR = "#2f7d32";
const CHART_COLORS = ["#0f5f74", "#1e7b96", "#2d97a8", "#78b6c1", "#b8dbe0", "#e2f0f2"];
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

const candidateHeaders = {
  txId: ["transaction_id", "txn_id", "id", "transactionid"],
  amount: ["amount", "transaction_amount", "amt", "value"],
  hour: ["hour", "transaction_hour", "time_hour", "tx_hour"],
  paymentMethod: ["payment_method", "paymentmethod", "method", "channel"],
  merchantCategory: ["merchant_category", "merchantcategory", "mcc", "category"],
  fraudLabel: ["is_fraud", "fraud", "label", "target", "fraud_flag", "class"]
};

function normalizeHeader(header) {
  return String(header || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^a-z0-9_]/g, "");
}

function getByCandidates(row, candidates) {
  const keys = Object.keys(row);
  const normalizedMap = new Map(keys.map((key) => [normalizeHeader(key), key]));

  for (const candidate of candidates) {
    if (normalizedMap.has(candidate)) {
      const actualKey = normalizedMap.get(candidate);
      return row[actualKey];
    }
  }

  return "";
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") {
    return NaN;
  }

  const numeric = Number(String(value).replace(/,/g, "").trim());
  return Number.isFinite(numeric) ? numeric : NaN;
}

function toFraudFlag(value) {
  const v = String(value ?? "")
    .trim()
    .toLowerCase();
  return ["1", "true", "yes", "y", "fraud", "positive"].includes(v);
}

function cleanText(value, fallback = "Unknown") {
  const text = String(value ?? "").trim();
  return text === "" ? fallback : text;
}

function toBinaryFlag(value) {
  const text = String(value ?? "")
    .trim()
    .toLowerCase();

  if (["1", "true", "yes", "y"].includes(text)) {
    return 1;
  }

  const numberValue = toNumber(value);
  if (Number.isFinite(numberValue) && numberValue > 0) {
    return 1;
  }

  return 0;
}

function toHour(value) {
  const n = toNumber(value);
  if (!Number.isFinite(n)) {
    return -1;
  }

  const intHour = Math.floor(n);
  return intHour >= 0 && intHour <= 23 ? intHour : -1;
}

function incrementCount(map, key) {
  map.set(key, (map.get(key) || 0) + 1);
}

function buildChartData(map, topN = 12) {
  return [...map.entries()]
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, topN);
}

function buildHourSeries(map) {
  const series = [];
  for (let hour = 0; hour <= 23; hour += 1) {
    series.push({ name: `${String(hour).padStart(2, "0")}:00`, count: map.get(String(hour)) || 0 });
  }
  return series;
}

async function callFraudApi(record) {
  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ record })
  });

  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }

  const json = await response.json();
  return {
    fraud: json?.fraud === 1,
    reasons: Array.isArray(json?.reasons) ? json.reasons : []
  };
}

async function scoreRowsWithBackend(parsedRows) {
  const health = await fetch(`${API_BASE_URL}/health`, { method: "GET" });
  if (!health.ok) {
    throw new Error("Backend health check failed");
  }

  try {
    const bulkResponse = await fetch(`${API_BASE_URL}/predict-bulk`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ records: parsedRows })
    });

    if (!bulkResponse.ok) {
      throw new Error(`HTTP ${bulkResponse.status}`);
    }

    const bulkJson = await bulkResponse.json();
    const predictions = Array.isArray(bulkJson?.predictions) ? bulkJson.predictions : [];

    return parsedRows.map((_, index) => ({
      fraud: predictions[index]?.fraud === 1,
      reasons: Array.isArray(predictions[index]?.reasons) ? predictions[index].reasons : []
    }));
  } catch (_bulkError) {
    const settled = await Promise.allSettled(parsedRows.map((row) => callFraudApi(row)));
    return settled.map((result) =>
      result.status === "fulfilled" ? result.value : { fraud: null, reasons: [] }
    );
  }
}

function analyzeRows(parsedRows, backendPredictions = []) {
  if (!parsedRows.length) {
    return null;
  }

  const headers = Object.keys(parsedRows[0]);
  const totalRecords = parsedRows.length;
  const totalFields = totalRecords * Math.max(headers.length, 1);
  let missingFields = 0;

  const rowFingerprints = new Map();
  let duplicateRecords = 0;

  const byHour = new Map();
  const byPayment = new Map();
  const byMerchant = new Map();

  const flagged = [];
  let fraudCount = 0;
  let amountSum = 0;
  let amountCount = 0;
  let maxAmount = 0;
  let backendScoredRows = 0;

  for (let i = 0; i < parsedRows.length; i += 1) {
    const row = parsedRows[i];

    for (const key of headers) {
      const value = row[key];
      if (String(value ?? "").trim() === "") {
        missingFields += 1;
      }
    }

    const normalizedFingerprint = headers
      .map((h) => `${normalizeHeader(h)}:${String(row[h] ?? "").trim().toLowerCase()}`)
      .join("|");

    if (rowFingerprints.has(normalizedFingerprint)) {
      duplicateRecords += 1;
    }
    rowFingerprints.set(normalizedFingerprint, true);

    const txId = cleanText(getByCandidates(row, candidateHeaders.txId), `TX-${i + 1}`);
    const amount = toNumber(getByCandidates(row, candidateHeaders.amount));
    if (Number.isFinite(amount)) {
      amountSum += amount;
      amountCount += 1;
      maxAmount = Math.max(maxAmount, amount);
    }
    const hourRaw = getByCandidates(row, candidateHeaders.hour);
    const hour = toHour(hourRaw);
    const paymentMethod = cleanText(getByCandidates(row, candidateHeaders.paymentMethod));
    const merchantCategory = cleanText(getByCandidates(row, candidateHeaders.merchantCategory));
    const hasFraudColumn = candidateHeaders.fraudLabel.some((name) =>
      Object.keys(row).some((key) => normalizeHeader(key) === name)
    );
    const labelFraud = toFraudFlag(getByCandidates(row, candidateHeaders.fraudLabel));
    const backendPrediction = backendPredictions[i];
    const backendFraud = typeof backendPrediction?.fraud === "boolean" ? backendPrediction.fraud : null;
    const backendReasons = Array.isArray(backendPrediction?.reasons)
      ? backendPrediction.reasons.filter(Boolean)
      : [];

    if (backendFraud !== null) {
      backendScoredRows += 1;
    }

    const reasons = [];
    if (Number.isFinite(amount) && amount >= 1000) {
      reasons.push("High amount (>= 1000)");
    }
    if (hour >= 0 && (hour <= 5 || hour >= 22)) {
      reasons.push("Odd-hour activity");
    }
    if (["gift_card", "crypto", "wire_transfer"].includes(normalizeHeader(paymentMethod))) {
      reasons.push("High-risk payment method");
    }
    if (["electronics", "gambling", "digital_goods"].includes(normalizeHeader(merchantCategory))) {
      reasons.push("High-risk merchant category");
    }

    const heuristicFraud = reasons.length >= 2;
    const isFraud =
      backendFraud !== null ? backendFraud : hasFraudColumn ? labelFraud : heuristicFraud;

    if (isFraud) {
      fraudCount += 1;
      flagged.push({
        txId,
        amount: Number.isFinite(amount) ? amount : null,
        hour: hour >= 0 ? hour : null,
        paymentMethod,
        merchantCategory,
        explanation:
          backendReasons.length > 0
            ? backendReasons.join("; ")
            : reasons.length
              ? reasons.join("; ")
              : "Model/label marked as fraud"
      });
    }

    incrementCount(byHour, hour >= 0 ? String(hour) : "Unknown");
    incrementCount(byPayment, paymentMethod);
    incrementCount(byMerchant, merchantCategory);
  }

  return {
    overview: {
      averageAmount: amountCount ? amountSum / amountCount : 0,
      maxAmount
    },
    quality: {
      totalRecords,
      totalFields,
      missingFields,
      missingRate: totalFields ? (missingFields / totalFields) * 100 : 0,
      duplicateRecords,
      duplicateRate: totalRecords ? (duplicateRecords / totalRecords) * 100 : 0
    },
    fraud: {
      fraudCount,
      fraudRate: totalRecords ? (fraudCount / totalRecords) * 100 : 0
    },
    backend: {
      scoredRows: backendScoredRows,
      scoreCoverage: totalRecords ? (backendScoredRows / totalRecords) * 100 : 0
    },
    charts: {
      hour: buildHourSeries(byHour),
      paymentMethod: buildChartData(byPayment, 10),
      merchantCategory: buildChartData(byMerchant, 10)
    },
    flagged: flagged
      .sort((a, b) => (b.amount || 0) - (a.amount || 0))
      .slice(0, 100)
  };
}

function formatPercent(value) {
  return `${value.toFixed(2)}%`;
}

function formatAmount(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2
  }).format(value);
}

function SectionCard({ title, children, className = "" }) {
  return (
    <section className={`section-card ${className}`}>
      <h2>{title}</h2>
      {children}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Kpi({ label, value, tone = "default" }) {
  return (
    <div className={`kpi kpi-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function FraudSplitChart({ fraudCount, totalRecords }) {
  const safeCount = Math.max(0, totalRecords - fraudCount);
  const data = [
    { name: "Fraud", value: fraudCount, color: FRAUD_COLOR },
    { name: "Safe", value: safeCount, color: SAFE_COLOR }
  ];

  return (
    <SectionCard title="Fraud Composition">
      <div className="donut-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <PieChart>
            <Pie data={data} dataKey="value" nameKey="name" innerRadius={62} outerRadius={95} paddingAngle={2}>
              {data.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="donut-legend">
        <span className="legend-dot fraud-dot" /> Fraud: {fraudCount}
        <span className="legend-dot safe-dot" /> Safe: {safeCount}
      </div>
    </SectionCard>
  );
}

function ChartBlock({ title, data, xKey = "name" }) {
  return (
    <SectionCard title={title}>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data} margin={{ top: 8, right: 8, left: -12, bottom: 40 }}>
            <CartesianGrid strokeDasharray="4 4" stroke="#d8d0c7" />
            <XAxis dataKey={xKey} angle={-25} textAnchor="end" interval={0} height={70} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Bar dataKey="count" name="Transactions" radius={[8, 8, 0, 0]}>
              {data.map((item, index) => (
                <Cell
                  key={item.name}
                  fill={item.name === "Unknown" ? "#b9ada1" : CHART_COLORS[index % CHART_COLORS.length]}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </SectionCard>
  );
}

function HourlyTrendChart({ data }) {
  return (
    <SectionCard title="Hourly Transaction Trend">
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -10, bottom: 8 }}>
            <defs>
              <linearGradient id="hourlyFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#1e7b96" stopOpacity={0.6} />
                <stop offset="100%" stopColor="#1e7b96" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="4 4" stroke="#d8d0c7" />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={1} />
            <YAxis allowDecimals={false} />
            <Tooltip />
            <Area type="monotone" dataKey="count" stroke="#0f5f74" fill="url(#hourlyFill)" strokeWidth={2} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </SectionCard>
  );
}

export default function App() {
  const [fileName, setFileName] = useState("");
  const [parseError, setParseError] = useState("");
  const [rows, setRows] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [backendPredictions] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [backendStatus, setBackendStatus] = useState("Backend not connected yet.");
  const [uploadResult, setUploadResult] = useState(null);

  const analysis = useMemo(() => analyzeRows(rows, backendPredictions), [rows, backendPredictions]);

  const onFileChange = (event) => {
    const file = event.target.files?.[0];
    setParseError("");
    setUploadResult(null);
    setBackendStatus("File selected. Click Upload & Detect Fraud.");

    if (!file) {
      setSelectedFile(null);
      return;
    }

    setFileName(file.name);
    setSelectedFile(file);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      dynamicTyping: false,
      complete: (result) => {
        if (result.errors?.length) {
          setParseError(result.errors[0].message || "Unable to parse CSV.");
          setRows([]);
          return;
        }

        const normalizedRows = result.data.filter((r) => Object.keys(r).length > 0);
        setRows(normalizedRows);
      },
      error: (error) => {
        setParseError(error.message || "Unable to parse CSV.");
        setRows([]);
      }
    });
  };

  const uploadFile = async () => {
    if (!selectedFile) {
      setParseError("Please choose a CSV file first.");
      return;
    }

    setIsUploading(true);
    setBackendStatus("Uploading file to ML backend...");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const res = await fetch(`${API_BASE_URL}/upload`, {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data?.error || `HTTP ${res.status}`);
      }

      setUploadResult(data);
      setBackendStatus(
        `Model prediction complete: ${data.fraud_detected} fraud / ${data.total_transactions} total.`
      );
    } catch (error) {
      setBackendStatus("Upload failed. Check backend server and CSV format.");
      setParseError(error.message || "Upload failed.");
      setUploadResult(null);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <main className="page">
      <header className="hero">
        <p className="eyebrow">Fraud Detection Analytics</p>
        <h1>
          Upload Your
          <span> Transaction Data</span>
        </h1>
        <p>
          Upload a CSV file to inspect data quality, review fraud exposure, and investigate
          suspicious transactions.
        </p>

        <div className="dropzone-wrap">
          <label className="upload-button" htmlFor="fileInput">
            Drag and drop your CSV
            <small>or click to browse your files</small>
          </label>
          <input id="fileInput" type="file" accept=".csv,text/csv" onChange={onFileChange} />
          <button className="detect-button" onClick={uploadFile} disabled={isUploading}>
            {isUploading ? "Detecting..." : "Upload & Detect Fraud"}
          </button>
          <p id="result" className="upload-result">
            {uploadResult
              ? `Total: ${uploadResult.total_transactions} | Fraud Detected: ${uploadResult.fraud_detected}`
              : "Result will appear here after upload."}
          </p>
          <span className="csv-pill">.CSV FILES ONLY</span>
        </div>

        <div className="expected-columns">
          <span>Expected CSV Columns</span>
          <div className="chip-row">
            <em>transaction_id</em>
            <em>amount</em>
            <em>hour</em>
            <em>payment_method</em>
            <em>merchant_category</em>
            <em>is_fraud</em>
          </div>
        </div>

        <div className="file-meta">
          {fileName ? <span>File: {fileName}</span> : <span>No CSV uploaded yet.</span>}
          <span className="backend-status">{backendStatus}</span>
          {isUploading ? <span className="backend-pending">Scoring transactions...</span> : null}
          {parseError ? <span className="error">Parse error: {parseError}</span> : null}
        </div>
      </header>

      {!analysis ? (
        <section className="empty">
          <h2>Waiting for data</h2>
          <p>Upload a CSV to generate quality checks, fraud summary, charts, and flagged rows.</p>
        </section>
      ) : (
        <>
          <div className="kpi-grid">
            <Kpi label="Transactions" value={analysis.quality.totalRecords} />
            <Kpi label="Fraud Rate" value={formatPercent(analysis.fraud.fraudRate)} tone="alert" />
            <Kpi label="Avg Transaction" value={formatAmount(analysis.overview.averageAmount)} />
            <Kpi label="Max Transaction" value={formatAmount(analysis.overview.maxAmount)} tone="accent" />
          </div>

          <div className="grid metrics-grid">
            <SectionCard title="Data Quality Report">
              <div className="metrics-list">
                <Metric label="Total Records" value={analysis.quality.totalRecords} />
                <Metric label="Columns" value={rows[0] ? Object.keys(rows[0]).length : 0} />
                <Metric label="Missing Fields" value={analysis.quality.missingFields} />
                <Metric label="Missing Rate" value={formatPercent(analysis.quality.missingRate)} />
                <Metric
                  label="Duplicate Records"
                  value={`${analysis.quality.duplicateRecords} (${formatPercent(
                    analysis.quality.duplicateRate
                  )})`}
                />
              </div>
            </SectionCard>

            <SectionCard title="Fraud Summary" className="fraud-summary">
              <div className="metrics-list">
                <Metric label="Fraud Count" value={analysis.fraud.fraudCount} />
                <Metric label="Fraud Percentage" value={formatPercent(analysis.fraud.fraudRate)} />
              </div>
              <div className="fraud-pill-row">
                <span className="pill fraud">Fraud: {analysis.fraud.fraudCount}</span>
                <span className="pill safe">
                  Non-Fraud: {analysis.quality.totalRecords - analysis.fraud.fraudCount}
                </span>
              </div>
              <div className="fraud-bar" aria-label="fraud distribution">
                <span
                  className="fraud-segment"
                  style={{ width: `${analysis.fraud.fraudRate}%`, backgroundColor: FRAUD_COLOR }}
                />
                <span
                  className="safe-segment"
                  style={{ width: `${100 - analysis.fraud.fraudRate}%`, backgroundColor: SAFE_COLOR }}
                />
              </div>
            </SectionCard>
          </div>

          <section className="notice-strip">
            {analysis.fraud.fraudCount ? (
              <span>Flagged transactions detected. Scroll for detailed explanations and categories.</span>
            ) : (
              <span>No fraud rows detected from current data and heuristics.</span>
            )}
          </section>

          <div className="grid charts-grid">
            <HourlyTrendChart data={analysis.charts.hour} />
            <FraudSplitChart
              fraudCount={analysis.fraud.fraudCount}
              totalRecords={analysis.quality.totalRecords}
            />
            <ChartBlock title="Transactions by Payment Method" data={analysis.charts.paymentMethod} />
            <ChartBlock
              title="Transactions by Merchant Category"
              data={analysis.charts.merchantCategory}
            />
          </div>

          <SectionCard title="Flagged Transactions with Explanations">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Transaction ID</th>
                    <th>Amount</th>
                    <th>Hour</th>
                    <th>Payment Method</th>
                    <th>Merchant Category</th>
                    <th>Explanation</th>
                  </tr>
                </thead>
                <tbody>
                  {analysis.flagged.length ? (
                    analysis.flagged.map((tx) => (
                      <tr key={`${tx.txId}-${tx.explanation}`}>
                        <td>{tx.txId}</td>
                        <td>{formatAmount(tx.amount)}</td>
                        <td>{tx.hour ?? "-"}</td>
                        <td>{tx.paymentMethod}</td>
                        <td>{tx.merchantCategory}</td>
                        <td>{tx.explanation}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6} className="empty-row">
                        No flagged transactions for the uploaded dataset.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </SectionCard>
        </>
      )}
    </main>
  );
}
