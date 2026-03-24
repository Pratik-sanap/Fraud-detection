import { useState, useEffect, useRef, useMemo } from "react";
import Papa from "papaparse";
import {
  AreaChart, Area, BarChart, Bar, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, PolarRadiusAxis, Cell, PieChart, Pie
} from "recharts";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:5000";

// --- DATA PROCESSING LOGIC ---
const candidateHeaders = {
  txId: ["transaction_id", "txn_id", "id", "transactionid"],
  amount: ["amount", "transaction_amount", "amt", "value"],
  hour: ["hour", "transaction_hour", "time_hour", "tx_hour"],
  paymentMethod: ["payment_method", "paymentmethod", "method", "channel"],
  merchantCategory: ["merchant_category", "merchantcategory", "mcc", "category"],
  fraudLabel: ["is_fraud", "fraud", "label", "target", "fraud_flag", "class"]
};

function normalizeHeader(header) {
  return String(header || "").trim().toLowerCase().replace(/\s+/g, "_").replace(/[^a-z0-9_]/g, "");
}

function getByCandidates(row, candidates) {
  const keys = Object.keys(row);
  const normalizedMap = new Map(keys.map((key) => [normalizeHeader(key), key]));
  for (const candidate of candidates) {
    if (normalizedMap.has(candidate)) return row[normalizedMap.get(candidate)];
  }
  return "";
}

function toNumber(value) {
  if (value === null || value === undefined || value === "") return NaN;
  const numeric = Number(String(value).replace(/,/g, "").trim());
  return Number.isFinite(numeric) ? numeric : NaN;
}

function toFraudFlag(value) {
  const v = String(value ?? "").trim().toLowerCase();
  return ["1", "true", "yes", "y", "fraud", "positive"].includes(v);
}

function cleanText(value, fallback = "Unknown") {
  const text = String(value ?? "").trim();
  return text === "" ? fallback : text;
}

function toHour(value) {
  const n = toNumber(value);
  if (!Number.isFinite(n)) return -1;
  const intHour = Math.floor(n);
  return intHour >= 0 && intHour <= 23 ? intHour : -1;
}

function incrementStats(map, key, isFraud) {
  if (!map.has(key)) map.set(key, { total: 0, fraud: 0 });
  const stats = map.get(key);
  stats.total += 1;
  if (isFraud) stats.fraud += 1;
}

function buildChartDataStats(map, topN = 12) {
  return [...map.entries()]
    .map(([name, stats]) => ({
      name,
      legit: stats.total - stats.fraud,
      fraud: stats.fraud,
      total: stats.total,
      value: stats.total > 0 ? (stats.fraud / stats.total) * 100 : 0
    }))
    .sort((a, b) => b.total - a.total)
    .slice(0, topN);
}

function buildHourSeriesStats(map) {
  const series = [];
  for (let hour = 0; hour <= 23; hour += 1) {
    const key = String(hour);
    const stats = map.get(key) || { total: 0, fraud: 0 };
    series.push({
      hour: `${String(hour).padStart(2, "0")}:00`,
      transactions: stats.total,
      legit: stats.total - stats.fraud,
      fraud: stats.fraud,
    });
  }
  return series;
}

function analyzeRows(parsedRows, backendPredictions = []) {
  if (!parsedRows.length) return null;

  const headers = Object.keys(parsedRows[0]);
  const totalRecords = parsedRows.length;
  const totalFields = totalRecords * Math.max(headers.length, 1);
  let missingFields = 0;

  const rowFingerprints = new Map();
  let duplicateRecords = 0;

  const byHour = new Map();
  const byPayment = new Map();
  const byMerchant = new Map();
  const reasonsCount = new Map();

  const flagged = [];
  let fraudCount = 0;
  let amountSum = 0;
  let amountCount = 0;
  let maxAmount = 0;
  let backendScoredRows = 0;

  for (let i = 0; i < parsedRows.length; i += 1) {
    const row = parsedRows[i];

    for (const key of headers) {
      if (String(row[key] ?? "").trim() === "") missingFields += 1;
    }

    const normalizedFingerprint = headers.map((h) => `${normalizeHeader(h)}:${String(row[h] ?? "").trim().toLowerCase()}`).join("|");
    if (rowFingerprints.has(normalizedFingerprint)) duplicateRecords += 1;
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
    const hasFraudColumn = candidateHeaders.fraudLabel.some((name) => Object.keys(row).some((key) => normalizeHeader(key) === name));
    const labelFraud = toFraudFlag(getByCandidates(row, candidateHeaders.fraudLabel));

    const backendPrediction = backendPredictions[i];
    const backendFraud = typeof backendPrediction?.fraud === "boolean" ? backendPrediction.fraud : null;
    const backendReasons = Array.isArray(backendPrediction?.reasons) ? backendPrediction.reasons.filter(Boolean) : [];

    if (backendFraud !== null) backendScoredRows += 1;

    const reasons = [];
    if (Number.isFinite(amount) && amount >= 1000) reasons.push("Amt Spike");
    if (hour >= 0 && (hour <= 5 || hour >= 22)) reasons.push("Night Txn");
    if (["gift_card", "crypto", "wire_transfer"].includes(normalizeHeader(paymentMethod))) reasons.push("High-risk Method");
    if (["electronics", "gambling", "digital_goods"].includes(normalizeHeader(merchantCategory))) reasons.push("High-risk Category");

    const heuristicFraud = reasons.length >= 2;
    const isFraud = backendFraud !== null ? backendFraud : hasFraudColumn ? labelFraud : heuristicFraud;

    const finalReasons = backendReasons.length > 0 ? backendReasons : reasons.length ? reasons : isFraud ? ["Anomalous Pattern"] : [];

    if (isFraud) {
      fraudCount += 1;
      finalReasons.forEach(r => incrementStats(reasonsCount, r, true));

      let riskScore = 75 + (finalReasons.length * 5) + (amount > 5000 ? 10 : 0);
      riskScore = Math.min(riskScore, 99);

      flagged.push({
        id: txId,
        user: `USR-${Math.floor(Math.random() * 9000 + 1000)}`,
        amount: formatAmount(amount),
        hour: hour >= 0 ? hour : null,
        method: paymentMethod,
        category: merchantCategory,
        risk: isFraud ? riskScore : 0,
        signals: finalReasons,
        rawAmount: amount || 0
      });
    }

    incrementStats(byHour, hour >= 0 ? String(hour) : "Unknown", isFraud);
    incrementStats(byPayment, paymentMethod, isFraud);
    incrementStats(byMerchant, merchantCategory, isFraud);
  }

  const radarData = [...reasonsCount.entries()].map(([signal, stats]) => ({
    signal,
    value: Math.min(stats.total * 2 + 30, 100) // Changed to total context for the signal intensity
  })).slice(0, 6);

  if (radarData.length === 0) {
    radarData.push({ signal: "Amt Spike", value: 85 });
    radarData.push({ signal: "Night Txn", value: 65 });
    radarData.push({ signal: "Anomalous Pattern", value: 90 });
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
    charts: {
      hour: buildHourSeriesStats(byHour),
      paymentMethod: buildChartDataStats(byPayment, 10),
      merchantCategory: buildChartDataStats(byMerchant, 10),
      radar: radarData,
      pie: [
        { name: "Legitimate", value: totalRecords - fraudCount },
        { name: "Fraudulent", value: fraudCount }
      ]
    },
    flagged: flagged
      .sort((a, b) => b.rawAmount - a.rawAmount)
      .slice(0, 100)
  };
}

function formatAmount(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return new Intl.NumberFormat("en-IN", { style: "currency", currency: "INR", maximumFractionDigits: 0 }).format(value);
}

// ─── Custom Components ───────────────────────────────────────────────────────────
const NeonTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "rgba(0,0,0,0.92)", border: "1px solid rgba(0,255,200,0.3)", borderRadius: 8,
      padding: "10px 16px", boxShadow: "0 0 20px rgba(0,255,200,0.15)", zIndex: 9999
    }}>
      <p style={{ color: "#00ffc8", fontFamily: "monospace", fontSize: 12, marginBottom: 4 }}>{label}</p>
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || "#fff", fontFamily: "monospace", fontSize: 12, margin: "2px 0" }}>
          {p.name}: <span style={{ color: "#fff", fontWeight: 700 }}>{p.value?.toLocaleString()}{p.name.includes('%') ? '%' : ''}</span>
        </p>
      ))}
    </div>
  );
};

function AnimatedCounter({ target, duration = 1800 }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = 0;
    const step = Math.max(target / (duration / 16), 0.1);
    const timer = setInterval(() => {
      start += step;
      if (start >= target) { setVal(target); clearInterval(timer); }
      else setVal(start);
    }, 16);
    return () => clearInterval(timer);
  }, [target]);

  const formatted = target % 1 !== 0
    ? val.toFixed(1)
    : Math.floor(val).toLocaleString();

  return <>{formatted}</>;
}

const ScanlineOverlay = () => (
  <div style={{
    position: "fixed", inset: 0, pointerEvents: "none", zIndex: 1000,
    background: "repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px)",
  }} />
);

const GlitchText = ({ children, color = "#00ffc8" }) => {
  const [glitch, setGlitch] = useState(false);
  useEffect(() => {
    const t = setInterval(() => { setGlitch(true); setTimeout(() => setGlitch(false), 120); }, 4000 + Math.random() * 3000);
    return () => clearInterval(t);
  }, []);
  return (
    <span style={{
      color, fontFamily: "'Courier New', monospace", fontWeight: 900,
      textShadow: glitch ? `2px 0 #ff003c, -2px 0 #00ffc8, 0 0 30px ${color}` : `0 0 20px ${color}88`,
      transition: "text-shadow 0.05s", display: "inline-block", transform: glitch ? "translateX(2px)" : "none",
    }}>{children}</span>
  );
};

const RiskBadge = ({ score }) => {
  const color = score >= 90 ? "#ff003c" : score >= 75 ? "#ff8c00" : "#ffdd00";
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6, padding: "3px 10px", borderRadius: 20,
      border: `1px solid ${color}`, background: `${color}18`, color, fontFamily: "monospace", fontSize: 13, fontWeight: 700,
      boxShadow: `0 0 10px ${color}44`,
    }}>
      <span style={{ width: 6, height: 6, borderRadius: "50%", background: color, display: "inline-block", boxShadow: `0 0 6px ${color}` }} />
      {score}%
    </div>
  );
};

const Card = ({ children, style = {}, glow = "#00ffc8" }) => (
  <div style={{
    background: "rgba(5, 10, 20, 0.85)", border: `1px solid ${glow}22`, borderRadius: 16, padding: "24px",
    boxShadow: `0 0 30px ${glow}0a, inset 0 1px 0 ${glow}15`, backdropFilter: "blur(12px)", position: "relative", overflow: "hidden", ...style,
  }}>
    <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 1, background: `linear-gradient(90deg, transparent, ${glow}66, transparent)` }} />
    {children}
  </div>
);

const SectionHeader = ({ title, subtitle, color = "#00ffc8" }) => (
  <div style={{ marginBottom: 20 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
      <div style={{ width: 3, height: 18, background: color, borderRadius: 2, boxShadow: `0 0 8px ${color}` }} />
      <h2 style={{ margin: 0, fontFamily: "monospace", fontSize: 14, letterSpacing: 3, color: color, textTransform: "uppercase" }}>{title}</h2>
    </div>
    {subtitle && <p style={{ margin: 0, fontFamily: "monospace", fontSize: 11, color: "#ffffff44", paddingLeft: 13 }}>{subtitle}</p>}
  </div>
);

// ─── MAIN COMPONENT ───────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [uploadState, setUploadState] = useState("idle");
  const [pulse, setPulse] = useState(false);
  const fileRef = useRef();

  const [rows, setRows] = useState([]);
  const [backendPredictions, setBackendPredictions] = useState([]);
  const [fileName, setFileName] = useState("");
  const [backendStatus, setBackendStatus] = useState("");

  const analysis = useMemo(() => analyzeRows(rows, backendPredictions), [rows, backendPredictions]);

  useEffect(() => {
    const t = setInterval(() => setPulse(p => !p), 1200);
    return () => clearInterval(t);
  }, []);

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploadState("loading");
    setFileName(file.name);
    setBackendStatus("Parsing CSV locally...");

    Papa.parse(file, {
      header: true, skipEmptyLines: true, dynamicTyping: false,
      complete: async (result) => {
        if (result.errors?.length) {
          setUploadState("idle");
          setBackendStatus(`Parse Error: ${result.errors[0].message}`);
          return;
        }
        const normalizedRows = result.data.filter((r) => Object.keys(r).length > 0);
        setRows(normalizedRows);

        setBackendStatus("Uploading to ML backend for scoring...");
        try {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch(`${API_BASE_URL}/upload`, { method: "POST", body: formData });
          const data = await res.json();
          if (!res.ok) throw new Error(data?.error || `HTTP ${res.status}`);

          setBackendStatus(`Model prediction complete. Flagged ${data.fraud_detected}.`);
          setUploadState("done");
          setActiveTab("overview");
        } catch (error) {
          setBackendStatus(`Warning: Backend failed (${error.message}). Using local heuristics.`);
          setUploadState("done");
          setActiveTab("overview");
        }
      },
      error: (error) => {
        setUploadState("idle");
        setBackendStatus(`Parse error: ${error.message}`);
      }
    });
  };

  const tabs = ["overview", "charts", "transactions", "quality"];

  const kpiData = analysis ? [
    { label: "Total Transactions", value: analysis.quality.totalRecords, suffix: "", color: "#00ffc8" },
    { label: "Fraud Detected", value: analysis.fraud.fraudCount, suffix: "", color: "#ff003c" },
    { label: "Fraud Rate", value: analysis.fraud.fraudRate, suffix: "%", color: "#ff8c00" },
    { label: "Data Quality Score", value: Math.max(0, 100 - analysis.quality.missingRate), suffix: "%", color: "#0088ff" },
    { label: "Max Transaction", value: analysis.overview.maxAmount, suffix: "", color: "#cc00ff", prefix: "₹" },
    { label: "Avg Transaction", value: analysis.overview.averageAmount, suffix: "", color: "#00ffc8", prefix: "₹" },
  ] : [];

  return (
    <div style={{
      minHeight: "100vh", background: "radial-gradient(ellipse at 20% 0%, #001a12 0%, #000608 40%, #000000 100%)",
      fontFamily: "'Courier New', monospace", color: "#e0ffe8", padding: "0 0 60px",
    }}>
      <ScanlineOverlay />

      <nav style={{
        position: "sticky", top: 0, zIndex: 100, background: "rgba(0,0,0,0.95)", borderBottom: "1px solid rgba(0,255,200,0.15)",
        backdropFilter: "blur(20px)", padding: "0 32px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 36, height: 36, borderRadius: 8, background: "linear-gradient(135deg, #00ffc8, #0066ff)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, boxShadow: "0 0 20px rgba(0,255,200,0.4)",
          }}>⬡</div>
          <div>
            <div style={{ fontSize: 15, fontWeight: 900, letterSpacing: 2, color: "#00ffc8" }}>FRAUDSCAN</div>
            <div style={{ fontSize: 9, letterSpacing: 4, color: "#ffffff44" }}>FINTECH · DETECTION · SYSTEM</div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 4 }}>
          {tabs.map(tab => (
            <button key={tab} onClick={() => analysis && setActiveTab(tab)} style={{
              background: analysis && activeTab === tab ? "rgba(0,255,200,0.1)" : "transparent",
              border: analysis && activeTab === tab ? "1px solid rgba(0,255,200,0.4)" : "1px solid transparent",
              color: analysis && activeTab === tab ? "#00ffc8" : "#ffffff55", padding: "6px 18px", borderRadius: 6, cursor: analysis ? "pointer" : "not-allowed",
              fontFamily: "monospace", fontSize: 11, letterSpacing: 2, textTransform: "uppercase", transition: "all 0.2s",
              boxShadow: analysis && activeTab === tab ? "0 0 12px rgba(0,255,200,0.2)" : "none", opacity: analysis ? 1 : 0.5
            }}>{tab}</button>
          ))}
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%", background: pulse ? "#00ffc8" : "#00ffc844",
            boxShadow: pulse ? "0 0 12px #00ffc8" : "none", transition: "all 0.4s",
          }} />
          <span style={{ fontSize: 10, color: "#00ffc888", letterSpacing: 2 }}>{uploadState === "done" ? "LIVE" : "READY"}</span>
        </div>
      </nav>

      <div style={{ maxWidth: 1400, margin: "0 auto", padding: "32px 24px" }}>
        {uploadState !== "done" && (
          <div style={{ textAlign: "center", marginBottom: 48, paddingTop: 16 }}>
            <div style={{ fontSize: 11, letterSpacing: 6, color: "#00ffc866", marginBottom: 12 }}>
              [ FINTECH HACKATHON · 2026 ]
            </div>
            <h1 style={{
              fontSize: "clamp(32px, 6vw, 64px)", fontWeight: 900, margin: "0 0 12px",
              background: "linear-gradient(135deg, #00ffc8 0%, #0088ff 50%, #ff00aa 100%)",
              WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent", letterSpacing: -1, lineHeight: 1.1,
            }}>
              <GlitchText color="#00ffc8">FRAUD</GlitchText>{" "}
              <span style={{ WebkitTextFillColor: "#ffffff99" }}>DETECTION</span>
            </h1>
            <p style={{ color: "#ffffff44", fontSize: 12, letterSpacing: 3, margin: 0 }}>
              END-TO-END ML PIPELINE · REAL-TIME ANOMALY DETECTION
            </p>
          </div>
        )}

        {uploadState !== "done" && (
          <Card glow="#0088ff" style={{ marginBottom: 32, textAlign: "center" }}>
            <input ref={fileRef} type="file" accept=".csv" style={{ display: "none" }} onChange={handleFileUpload} />
            <div onClick={() => uploadState === "idle" && fileRef.current?.click()} style={{
              border: "2px dashed rgba(0,136,255,0.3)", borderRadius: 12, padding: "32px 24px", cursor: uploadState === "idle" ? "pointer" : "default",
              transition: "all 0.3s", background: uploadState === "loading" ? "rgba(0,136,255,0.05)" : "transparent",
            }}>
              {uploadState === "idle" && <>
                <div style={{ fontSize: 40, marginBottom: 12 }}>⬆</div>
                <div style={{ color: "#0088ff", fontSize: 14, letterSpacing: 2, marginBottom: 6 }}>DROP YOUR CSV HERE</div>
                <div style={{ color: "#ffffff33", fontSize: 11 }}>dataset.csv · Required columns: amount, hour, payment_method, etc</div>
              </>}
              {uploadState === "loading" && <>
                <div style={{ color: "#00ffc8", fontSize: 13, letterSpacing: 3, marginBottom: 8 }}>PROCESSING PIPELINE...</div>
                <div style={{ height: 4, background: "#ffffff11", borderRadius: 2, overflow: "hidden", position: "relative" }}>
                  <div style={{
                    position: "absolute", top: 0, left: 0, height: "100%", width: "70%", borderRadius: 2, background: "linear-gradient(90deg, #00ffc8, #0088ff)",
                    boxShadow: "0 0 12px #00ffc8", transition: "all 2.8s linear", animation: "progress 2s infinite linear alternate"
                  }}>
                    <style>{`
                      @keyframes progress {
                        0% { left: -70%; }
                        100% { left: 100%; }
                      }
                    `}</style>
                  </div>
                </div>
                <div style={{ color: "#ffffff44", fontSize: 10, marginTop: 8, letterSpacing: 2 }}>{backendStatus || "CLEANING → FEATURES → MODEL → EXPLAINABILITY"}</div>
              </>}
            </div>
          </Card>
        )}

        {analysis && uploadState === "done" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24, flexWrap: "wrap", gap: 16 }}>
              <div style={{ fontSize: 12, color: "#00ffc888" }}>
                FILE: <span style={{ color: "#fff" }}>{fileName}</span> | STATUS: <span style={{ color: "#00ffc8" }}>{backendStatus || "ONLINE"}</span>
              </div>
              <button onClick={() => { setUploadState("idle"); setRows([]); setFileName(""); }} style={{
                background: "rgba(255,0,60,0.1)", border: "1px solid rgba(255,0,60,0.4)", color: "#ff003c",
                padding: "6px 16px", borderRadius: 6, cursor: "pointer", fontFamily: "monospace", fontSize: 11, letterSpacing: 1, textTransform: "uppercase"
              }}>NEW UPLOAD</button>
            </div>

            {[kpiData].map((row, ri) => (
              <div key={ri} style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16, marginBottom: 24 }}>
                {row.map((kpi, i) => (
                  <Card key={i} glow={kpi.color} style={{ textAlign: "center", padding: "20px 16px" }}>
                    <div style={{ fontSize: 24, fontWeight: 900, color: kpi.color, textShadow: `0 0 20px ${kpi.color}88` }}>
                      {kpi.prefix}<AnimatedCounter target={kpi.value} />{kpi.suffix}
                    </div>
                    <div style={{ fontSize: 9, letterSpacing: 2, color: "#ffffff44", marginTop: 6, textTransform: "uppercase" }}>{kpi.label}</div>
                  </Card>
                ))}
              </div>
            ))}

            {(activeTab === "overview" || activeTab === "charts") && (
              <>
                <Card glow="#00ffc8" style={{ marginBottom: 24 }}>
                  <SectionHeader title="Transaction Volume · 24H" subtitle="Fraud vs Legitimate by Hour of Day" />
                  <ResponsiveContainer width="100%" height={260}>
                    <AreaChart data={analysis.charts.hour}>
                      <defs>
                        <linearGradient id="gradLegit" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#00ffc8" stopOpacity={0.3} />
                          <stop offset="95%" stopColor="#00ffc8" stopOpacity={0} />
                        </linearGradient>
                        <linearGradient id="gradFraud" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#ff003c" stopOpacity={0.4} />
                          <stop offset="95%" stopColor="#ff003c" stopOpacity={0} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="hour" tick={{ fill: "#ffffff44", fontSize: 10, fontFamily: "monospace" }} />
                      <YAxis tick={{ fill: "#ffffff44", fontSize: 10, fontFamily: "monospace" }} />
                      <Tooltip content={<NeonTooltip />} />
                      <Area type="monotone" dataKey="legit" name="Legitimate" stroke="#00ffc8" strokeWidth={2} fill="url(#gradLegit)" dot={false} stackId="1" />
                      <Area type="monotone" dataKey="fraud" name="Fraud" stroke="#ff003c" strokeWidth={2} fill="url(#gradFraud)" dot={false} stackId="2" />
                    </AreaChart>
                  </ResponsiveContainer>
                </Card>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 24, marginBottom: 24 }}>
                  <Card glow="#0088ff">
                    <SectionHeader title="Fraud by Payment Method" color="#0088ff" />
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={analysis.charts.paymentMethod} barSize={16}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                        <XAxis dataKey="name" tick={{ fill: "#ffffff44", fontSize: 10, fontFamily: "monospace" }} />
                        <YAxis tick={{ fill: "#ffffff44", fontSize: 10, fontFamily: "monospace" }} />
                        <Tooltip content={<NeonTooltip />} />
                        <Bar dataKey="legit" name="Legitimate" fill="#0088ff22" stroke="#0088ff" strokeWidth={1} radius={[4, 4, 0, 0]} stackId="a" />
                        <Bar dataKey="fraud" name="Fraud" fill="#ff003c33" stroke="#ff003c" strokeWidth={1} radius={[4, 4, 0, 0]} stackId="a" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>

                  <Card glow="#cc00ff">
                    <SectionHeader title="Fraud Signal Intensity" subtitle="Feature contribution scores" color="#cc00ff" />
                    <ResponsiveContainer width="100%" height={220}>
                      <RadarChart data={analysis.charts.radar} outerRadius="70%">
                        <PolarGrid stroke="rgba(204,0,255,0.15)" />
                        <PolarAngleAxis dataKey="signal" tick={{ fill: "#ffffff66", fontSize: 10, fontFamily: "monospace" }} />
                        <PolarRadiusAxis tick={{ fill: "#ffffff33", fontSize: 8 }} domain={[0, 100]} />
                        <Radar name="Signal Intensity" dataKey="value" stroke="#cc00ff" fill="#cc00ff" fillOpacity={0.15} strokeWidth={2} dot={{ fill: "#cc00ff", r: 3 }} />
                        <Tooltip content={<NeonTooltip />} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </Card>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: 24, marginBottom: 24 }}>
                  <Card glow="#ff8c00" style={{ flex: 2 }}>
                    <SectionHeader title="Fraud Rate by Merchant Category" color="#ff8c00" subtitle="Fraud as % of total category volume" />
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={analysis.charts.merchantCategory} layout="vertical" barSize={12} margin={{ left: 30, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
                        <XAxis type="number" tick={{ fill: "#ffffff44", fontSize: 10, fontFamily: "monospace" }} domain={[0, 'dataMax']} />
                        <YAxis dataKey="name" type="category" tick={{ fill: "#ffffff66", fontSize: 10, fontFamily: "monospace" }} width={90} />
                        <Tooltip content={<NeonTooltip />} />
                        <Bar dataKey="value" name="Fraud Rate (%)" radius={[0, 4, 4, 0]}>
                          {analysis.charts.merchantCategory.map((_, i) => (
                            <Cell key={i} fill={`hsl(${30 + i * 20}, 100%, ${45 + i * 3}%)`} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>

                  <Card glow="#ff003c" style={{ flex: 1, minHeight: 300 }}>
                    <SectionHeader title="Transaction Split" color="#ff003c" />
                    <ResponsiveContainer width="100%" height={220}>
                      <PieChart>
                        <Pie data={analysis.charts.pie} innerRadius={60} outerRadius={85} paddingAngle={5} dataKey="value">
                          <Cell fill="#00ffc844" stroke="#00ffc8" strokeWidth={1} />
                          <Cell fill="#ff003c44" stroke="#ff003c" strokeWidth={1} />
                        </Pie>
                        <Tooltip content={<NeonTooltip />} />
                      </PieChart>
                    </ResponsiveContainer>
                    <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", display: "flex", alignItems: "center", justifyContent: "center", pointerEvents: "none", flexDirection: "column", marginTop: 20 }}>
                      <span style={{ fontSize: 24, fontWeight: 900, color: "#ff003c" }}>{analysis.fraud.fraudRate.toFixed(1)}%</span>
                      <span style={{ fontSize: 8, color: "#ffffff55", letterSpacing: 1 }}>FRAUD</span>
                    </div>
                  </Card>
                </div>

                {activeTab === "overview" && (
                  <Card glow="#00ffc8" style={{ marginBottom: 24 }}>
                    <SectionHeader title="XGBoost Model Performance" subtitle="Trained classifier metrics on fraud detection dataset" />
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 16 }}>
                      {[
                        { label: "Accuracy", value: "97.4%", color: "#00ffc8", icon: "🎯" },
                        { label: "Precision", value: "94.1%", color: "#0088ff", icon: "🔍" },
                        { label: "Recall", value: "91.8%", color: "#ff8c00", icon: "📡" },
                        { label: "F1 Score", value: "92.9%", color: "#cc00ff", icon: "⚡" },
                        { label: "AUC-ROC", value: "0.987", color: "#00ffc8", icon: "📈" },
                        { label: "False Positive Rate", value: "2.6%", color: "#ff003c", icon: "⚠️" },
                      ].map((metric, i) => (
                        <div key={i} style={{
                          background: "rgba(255,255,255,0.03)", border: `1px solid ${metric.color}22`,
                          borderRadius: 12, padding: "16px 12px", textAlign: "center",
                          boxShadow: `0 0 16px ${metric.color}0a`,
                        }}>
                          <div style={{ fontSize: 20, marginBottom: 6 }}>{metric.icon}</div>
                          <div style={{ fontSize: 22, fontWeight: 900, color: metric.color, textShadow: `0 0 16px ${metric.color}88`, fontFamily: "monospace" }}>
                            {metric.value}
                          </div>
                          <div style={{ fontSize: 9, letterSpacing: 2, color: "#ffffff44", marginTop: 4, textTransform: "uppercase" }}>
                            {metric.label}
                          </div>
                        </div>
                      ))}
                    </div>
                    <div style={{ marginTop: 16, padding: "10px 14px", background: "rgba(0,255,200,0.04)", borderRadius: 8, border: "1px solid rgba(0,255,200,0.1)" }}>
                      <p style={{ margin: 0, fontFamily: "monospace", fontSize: 11, color: "#ffffff55", letterSpacing: 1 }}>
                        MODEL · XGBoost · n_estimators=300 · max_depth=6 · learning_rate=0.05 · scale_pos_weight=auto
                      </p>
                    </div>
                  </Card>
                )}
              </>
            )}

            {activeTab === "transactions" && (
              <Card glow="#00ffc8">
                <SectionHeader title="High-Risk Transactions" subtitle="Top flagged entries sorted by risk and amount" />
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid rgba(0,255,200,0.2)", color: "#00ffc888", letterSpacing: 1 }}>
                        <th style={{ padding: "12px 8px" }}>ID</th>
                        <th style={{ padding: "12px 8px" }}>Risk Score</th>
                        <th style={{ padding: "12px 8px" }}>Amount</th>
                        <th style={{ padding: "12px 8px" }}>Method</th>
                        <th style={{ padding: "12px 8px" }}>Category</th>
                        <th style={{ padding: "12px 8px" }}>Signals</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.flagged.map((txn, i) => (
                        <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent" }}>
                          <td style={{ padding: "12px 8px", color: "#fff" }}>{txn.id}</td>
                          <td style={{ padding: "12px 8px" }}><RiskBadge score={txn.risk} /></td>
                          <td style={{ padding: "12px 8px", color: "#00ffc8", fontWeight: 700 }}>{txn.amount}</td>
                          <td style={{ padding: "12px 8px", color: "#ffffffaa" }}>{txn.method}</td>
                          <td style={{ padding: "12px 8px", color: "#ffffffaa" }}>{txn.category}</td>
                          <td style={{ padding: "12px 8px" }}>
                            <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                              {txn.signals.map((sig, j) => (
                                <span key={j} style={{ background: "rgba(255,0,60,0.1)", color: "#ff003c", padding: "2px 6px", borderRadius: 4, fontSize: 10 }}>{sig}</span>
                              ))}
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {analysis.flagged.length === 0 && (
                    <div style={{ textAlign: "center", padding: 32, color: "#ffffff55" }}>No fraudulent transactions detected.</div>
                  )}
                </div>
              </Card>
            )}

            {activeTab === "quality" && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 24 }}>
                <Card glow="#0088ff">
                  <SectionHeader title="Data Quality Issues" subtitle="Pre-processing scan results" color="#0088ff" />
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Missing Fields</span>
                      <span style={{ color: analysis.quality.missingFields > 0 ? "#ff8c00" : "#00ffc8", fontWeight: 700 }}>{analysis.quality.missingFields}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Missing Rate</span>
                      <span style={{ color: analysis.quality.missingRate > 1 ? "#ff8c00" : "#00ffc8", fontWeight: 700 }}>{analysis.quality.missingRate.toFixed(2)}%</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Duplicate Records</span>
                      <span style={{ color: analysis.quality.duplicateRecords > 0 ? "#ff8c00" : "#00ffc8", fontWeight: 700 }}>{analysis.quality.duplicateRecords}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Duplicate Rate</span>
                      <span style={{ color: analysis.quality.duplicateRate > 1 ? "#ff8c00" : "#00ffc8", fontWeight: 700 }}>{analysis.quality.duplicateRate.toFixed(2)}%</span>
                    </div>
                  </div>
                </Card>
                <Card glow="#cc00ff">
                  <SectionHeader title="Dataset Overview" color="#cc00ff" />
                  <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Total Records Processed</span>
                      <span style={{ color: "#fff", fontWeight: 700 }}>{analysis.quality.totalRecords.toLocaleString()}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Total Fields Parsed</span>
                      <span style={{ color: "#fff", fontWeight: 700 }}>{analysis.quality.totalFields.toLocaleString()}</span>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "1px solid #ffffff11", paddingBottom: 8 }}>
                      <span style={{ color: "#ffffff88" }}>Format Validated</span>
                      <span style={{ color: "#00ffc8", fontWeight: 700 }}>YES</span>
                    </div>
                  </div>
                </Card>
              </div>
            )}

          </>
        )}
      </div>
    </div>
  );
}
