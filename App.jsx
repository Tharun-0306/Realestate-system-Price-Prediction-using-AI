import { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line, CartesianGrid, Area, AreaChart } from "recharts";

// ─── Mock API (replace with real fetch to http://localhost:8000) ──────────────
const mockPredict = async (data) => {
  await new Promise(r => setTimeout(r, 1800));
  const base = 200000 + data.sqft_living * 150 + data.bedrooms * 15000 + data.bathrooms * 12000 + data.grade * 20000 + data.condition * 10000 + data.waterfront * 200000 + data.view * 25000;
  const price = base + Math.random() * 30000 - 15000;
  return {
    predicted_price: Math.round(price / 1000) * 1000,
    price_low: Math.round(price * 0.92 / 1000) * 1000,
    price_high: Math.round(price * 1.08 / 1000) * 1000,
    price_per_sqft: Math.round(price / data.sqft_living),
    confidence: data.grade >= 7 ? "High" : "Medium",
    feature_importance: {
      "Location": 32,
      "Living Area": 28,
      "Grade & Condition": 18,
      "Bedrooms/Baths": 14,
      "Special Features": 8,
    },
    comparable_properties: [
      { address: "1234 Maple St", price: Math.round(price * 0.97 / 1000) * 1000, bedrooms: data.bedrooms, sqft: data.sqft_living - 50, days_on_market: 12, status: "Sold" },
      { address: "567 Oak Ave", price: Math.round(price * 1.03 / 1000) * 1000, bedrooms: data.bedrooms, sqft: data.sqft_living + 120, days_on_market: 8, status: "Active" },
      { address: "890 Pine Rd", price: Math.round(price * 0.94 / 1000) * 1000, bedrooms: data.bedrooms - 1, sqft: data.sqft_living - 200, days_on_market: 25, status: "Sold" },
      { address: "321 Cedar Ln", price: Math.round(price * 1.06 / 1000) * 1000, bedrooms: data.bedrooms + 1, sqft: data.sqft_living + 300, days_on_market: 3, status: "Active" },
    ]
  };
};

const marketData = [
  { month: "Jan", price: 445000, listings: 920 }, { month: "Feb", price: 438000, listings: 870 },
  { month: "Mar", price: 462000, listings: 1050 }, { month: "Apr", price: 478000, listings: 1180 },
  { month: "May", price: 495000, listings: 1320 }, { month: "Jun", price: 512000, listings: 1450 },
  { month: "Jul", price: 508000, listings: 1380 }, { month: "Aug", price: 499000, listings: 1290 },
  { month: "Sep", price: 487000, listings: 1120 }, { month: "Oct", price: 471000, listings: 980 },
  { month: "Nov", price: 458000, listings: 860 }, { month: "Dec", price: 452000, listings: 790 },
];

const fmt = (n) => n >= 1000000 ? `$${(n/1000000).toFixed(2)}M` : `$${(n/1000).toFixed(0)}K`;

// ─── Components ───────────────────────────────────────────────────────────────

const Slider = ({ label, name, min, max, step = 1, value, onChange, format }) => (
  <div style={{ marginBottom: "20px" }}>
    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
      <span style={{ fontSize: "12px", color: "#8B9BB4", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>{label}</span>
      <span style={{ fontSize: "14px", color: "#E8F0FF", fontWeight: 700, fontFamily: "'Space Mono', monospace" }}>
        {format ? format(value) : value}
      </span>
    </div>
    <div style={{ position: "relative" }}>
      <input
        type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(name, parseFloat(e.target.value))}
        style={{
          width: "100%", height: "4px", borderRadius: "2px", outline: "none", cursor: "pointer",
          background: `linear-gradient(to right, #4F7FFF ${((value-min)/(max-min))*100}%, #1E2A3E ${((value-min)/(max-min))*100}%)`,
          WebkitAppearance: "none", appearance: "none",
        }}
      />
    </div>
  </div>
);

const Toggle = ({ label, name, value, onChange }) => (
  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
    <span style={{ fontSize: "12px", color: "#8B9BB4", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600 }}>{label}</span>
    <div
      onClick={() => onChange(name, value ? 0 : 1)}
      style={{
        width: "44px", height: "24px", borderRadius: "12px", cursor: "pointer", position: "relative",
        background: value ? "linear-gradient(135deg, #4F7FFF, #7B5EFF)" : "#1E2A3E",
        border: "1px solid rgba(79,127,255,0.3)", transition: "all 0.3s",
      }}
    >
      <div style={{
        position: "absolute", top: "3px", left: value ? "22px" : "3px", width: "16px", height: "16px",
        borderRadius: "50%", background: "#fff", transition: "left 0.3s", boxShadow: "0 2px 6px rgba(0,0,0,0.3)"
      }} />
    </div>
  </div>
);

const PulsingDot = () => (
  <div style={{ display: "inline-block", position: "relative", width: 12, height: 12, marginRight: 8 }}>
    <div style={{
      width: 12, height: 12, borderRadius: "50%", background: "#4F7FFF",
      animation: "pulse 1.5s infinite", position: "absolute"
    }} />
    <div style={{
      width: 12, height: 12, borderRadius: "50%", background: "#4F7FFF",
      opacity: 0.4, animation: "pulseRing 1.5s infinite", position: "absolute"
    }} />
  </div>
);

export default function App() {
  const [tab, setTab] = useState("predict");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [form, setForm] = useState({
    bedrooms: 3, bathrooms: 2, sqft_living: 2000, sqft_lot: 6000,
    floors: 1, waterfront: 0, view: 0, condition: 3, grade: 7,
    sqft_above: 1800, sqft_basement: 200, yr_built: 1990, yr_renovated: 0,
    zipcode: 98001, lat: 47.5, long: -122.0, sqft_living15: 1800, sqft_lot15: 5000
  });

  const handleChange = (name, value) => setForm(p => ({ ...p, [name]: value }));

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await mockPredict(form);
      setResult(res);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ minHeight: "100vh", background: "#080E1A", color: "#E8F0FF", fontFamily: "'DM Sans', 'Segoe UI', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #080E1A; }
        ::-webkit-scrollbar-thumb { background: #1E2A3E; border-radius: 3px; }
        input[type=range]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px; border-radius: 50%; background: linear-gradient(135deg, #4F7FFF, #7B5EFF); cursor: pointer; box-shadow: 0 0 0 3px rgba(79,127,255,0.2); }
        @keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.1)} }
        @keyframes pulseRing { 0%{transform:scale(1);opacity:0.4} 100%{transform:scale(2.5);opacity:0} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
        @keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
        @keyframes spin { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
        .card { background: #0D1521; border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 24px; }
        .btn-primary { background: linear-gradient(135deg, #4F7FFF, #7B5EFF); border: none; color: white; cursor: pointer; border-radius: 12px; font-family: inherit; font-weight: 700; letter-spacing: 0.05em; transition: all 0.2s; }
        .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(79,127,255,0.4); }
        .btn-primary:active { transform: translateY(0); }
        .tab { background: none; border: none; color: #8B9BB4; cursor: pointer; font-family: inherit; font-size: 14px; font-weight: 600; padding: 10px 20px; border-radius: 10px; transition: all 0.2s; }
        .tab.active { background: rgba(79,127,255,0.15); color: #4F7FFF; }
        .tag { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 0.06em; }
      `}</style>

      {/* Header */}
      <header style={{
        padding: "20px 40px", display: "flex", alignItems: "center", justifyContent: "space-between",
        borderBottom: "1px solid rgba(255,255,255,0.05)",
        background: "linear-gradient(180deg, rgba(79,127,255,0.05) 0%, transparent 100%)"
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div style={{
            width: 40, height: 40, borderRadius: 12,
            background: "linear-gradient(135deg, #4F7FFF, #7B5EFF)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20
          }}>🏠</div>
          <div>
            <div style={{ fontSize: 18, fontWeight: 800, letterSpacing: "-0.02em" }}>ProphetAI</div>
            <div style={{ fontSize: 11, color: "#8B9BB4", letterSpacing: "0.06em" }}>REAL ESTATE INTELLIGENCE</div>
          </div>
        </div>
        <nav style={{ display: "flex", gap: 4, background: "rgba(255,255,255,0.03)", borderRadius: 14, padding: 4 }}>
          {[["predict","🎯 Predict"], ["market","📈 Market"], ["about","⚙️ Model"]].map(([id, label]) => (
            <button key={id} className={`tab ${tab === id ? "active" : ""}`} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "#4F7FFF" }}>
          <PulsingDot /><span style={{ fontFamily: "'Space Mono', monospace" }}>94.2% accuracy</span>
        </div>
      </header>

      <main style={{ maxWidth: 1280, margin: "0 auto", padding: "40px 40px" }}>

        {/* ── PREDICT TAB ─────────────────────────────────────────────────── */}
        {tab === "predict" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24, animation: "fadeUp 0.5s ease" }}>

            {/* Form */}
            <div className="card">
              <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 28, letterSpacing: "-0.02em" }}>
                Property Details
                <div style={{ width: 40, height: 3, background: "linear-gradient(90deg, #4F7FFF, #7B5EFF)", borderRadius: 2, marginTop: 8 }} />
              </h2>

              <Slider label="Bedrooms" name="bedrooms" min={1} max={8} value={form.bedrooms} onChange={handleChange} />
              <Slider label="Bathrooms" name="bathrooms" min={1} max={6} step={0.5} value={form.bathrooms} onChange={handleChange} />
              <Slider label="Living Area (sqft)" name="sqft_living" min={500} max={8000} step={50} value={form.sqft_living} onChange={handleChange} format={v => `${v.toLocaleString()} ft²`} />
              <Slider label="Lot Size (sqft)" name="sqft_lot" min={1000} max={40000} step={500} value={form.sqft_lot} onChange={handleChange} format={v => `${v.toLocaleString()} ft²`} />
              <Slider label="Floors" name="floors" min={1} max={3} step={0.5} value={form.floors} onChange={handleChange} />

              <div style={{ borderTop: "1px solid rgba(255,255,255,0.05)", margin: "20px 0 20px" }} />

              <Slider label="Building Grade (1–13)" name="grade" min={1} max={13} value={form.grade} onChange={handleChange} format={v => `${v}/13 — ${["Poor","Below Avg","Below Avg","Average","Average","Average","Average","Good","Good","Better","Very Good","Excellent","Luxury","Mansion"][v-1] || ""}`} />
              <Slider label="Condition (1–5)" name="condition" min={1} max={5} value={form.condition} onChange={handleChange} format={v => ["Poor","Fair","Average","Good","Excellent"][v-1]} />
              <Slider label="View Quality (0–4)" name="view" min={0} max={4} value={form.view} onChange={handleChange} format={v => ["None","Fair","Average","Good","Excellent"][v]} />
              <Slider label="Year Built" name="yr_built" min={1900} max={2024} value={form.yr_built} onChange={handleChange} />
              <Toggle label="Waterfront Property" name="waterfront" value={form.waterfront} onChange={handleChange} />

              <button className="btn-primary" onClick={handlePredict} disabled={loading}
                style={{ width: "100%", padding: "16px", fontSize: 15, marginTop: 8 }}>
                {loading ? "⏳ Analyzing Property..." : "🔮 Predict Price"}
              </button>
            </div>

            {/* Results */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {loading && (
                <div className="card" style={{ textAlign: "center", padding: "60px 24px" }}>
                  <div style={{ fontSize: 48, marginBottom: 20, animation: "spin 2s linear infinite", display: "inline-block" }}>🔮</div>
                  <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>Analyzing Property...</div>
                  <div style={{ color: "#8B9BB4", fontSize: 14 }}>Running XGBoost prediction model</div>
                  <div style={{
                    marginTop: 24, height: 4, borderRadius: 2,
                    background: "linear-gradient(90deg, #4F7FFF 30%, transparent 100%)",
                    backgroundSize: "200% 100%", animation: "shimmer 1.5s infinite"
                  }} />
                </div>
              )}

              {!result && !loading && (
                <div className="card" style={{ textAlign: "center", padding: "60px 24px", border: "2px dashed rgba(79,127,255,0.2)" }}>
                  <div style={{ fontSize: 64, marginBottom: 20 }}>🏡</div>
                  <div style={{ fontSize: 18, fontWeight: 700, marginBottom: 8 }}>Ready to Predict</div>
                  <div style={{ color: "#8B9BB4", fontSize: 14, lineHeight: 1.7 }}>
                    Adjust the property details on the left<br />and click Predict Price
                  </div>
                </div>
              )}

              {result && !loading && (
                <>
                  {/* Main Price Card */}
                  <div className="card" style={{
                    background: "linear-gradient(135deg, #0D1521 0%, #111B2E 100%)",
                    border: "1px solid rgba(79,127,255,0.25)", animation: "fadeUp 0.4s ease"
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div>
                        <div style={{ fontSize: 12, color: "#8B9BB4", textTransform: "uppercase", letterSpacing: "0.1em", fontWeight: 600, marginBottom: 6 }}>Estimated Value</div>
                        <div style={{ fontSize: 48, fontWeight: 900, letterSpacing: "-0.03em", fontFamily: "'Space Mono', monospace", background: "linear-gradient(135deg, #4F7FFF, #A78BFF)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
                          {fmt(result.predicted_price)}
                        </div>
                        <div style={{ fontSize: 14, color: "#8B9BB4", marginTop: 6 }}>
                          {fmt(result.price_low)} — {fmt(result.price_high)} range
                        </div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <span className="tag" style={{ background: result.confidence === "High" ? "rgba(34,197,94,0.15)" : "rgba(251,191,36,0.15)", color: result.confidence === "High" ? "#22C55E" : "#FBbf24" }}>
                          {result.confidence === "High" ? "✓" : "~"} {result.confidence} Confidence
                        </span>
                        <div style={{ marginTop: 12, fontFamily: "'Space Mono', monospace", fontSize: 13 }}>
                          <span style={{ color: "#8B9BB4" }}>$/sqft </span>
                          <span style={{ color: "#E8F0FF", fontWeight: 700 }}>${result.price_per_sqft}</span>
                        </div>
                      </div>
                    </div>

                    {/* Price bar */}
                    <div style={{ marginTop: 20, background: "#080E1A", borderRadius: 8, height: 8, position: "relative", overflow: "hidden" }}>
                      <div style={{
                        position: "absolute", left: "8%", right: "8%", top: 0, bottom: 0,
                        background: "rgba(79,127,255,0.3)", borderRadius: 8
                      }} />
                      <div style={{
                        position: "absolute", left: "50%", top: -2, bottom: -2, width: 4,
                        background: "linear-gradient(180deg, #4F7FFF, #7B5EFF)", borderRadius: 2,
                        transform: "translateX(-50%)"
                      }} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 11, color: "#8B9BB4" }}>
                      <span>{fmt(result.price_low)}</span><span style={{ color: "#4F7FFF", fontWeight: 700 }}>{fmt(result.predicted_price)}</span><span>{fmt(result.price_high)}</span>
                    </div>
                  </div>

                  {/* Feature Importance */}
                  <div className="card" style={{ animation: "fadeUp 0.5s ease" }}>
                    <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#8B9BB4", marginBottom: 20 }}>Price Drivers</h3>
                    {Object.entries(result.feature_importance).map(([key, val]) => (
                      <div key={key} style={{ marginBottom: 14 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6, fontSize: 13 }}>
                          <span>{key}</span>
                          <span style={{ color: "#4F7FFF", fontFamily: "'Space Mono', monospace", fontWeight: 700 }}>{val}%</span>
                        </div>
                        <div style={{ height: 6, background: "#1E2A3E", borderRadius: 3, overflow: "hidden" }}>
                          <div style={{ width: `${val}%`, height: "100%", background: "linear-gradient(90deg, #4F7FFF, #7B5EFF)", borderRadius: 3, transition: "width 1s ease" }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Comparables */}
                  <div className="card" style={{ animation: "fadeUp 0.6s ease" }}>
                    <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#8B9BB4", marginBottom: 20 }}>Comparable Properties</h3>
                    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                      {result.comparable_properties.map((c, i) => (
                        <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 16px", background: "#080E1A", borderRadius: 10, border: "1px solid rgba(255,255,255,0.04)" }}>
                          <div>
                            <div style={{ fontSize: 13, fontWeight: 600 }}>{c.address}</div>
                            <div style={{ fontSize: 11, color: "#8B9BB4", marginTop: 2 }}>{c.bedrooms} bd · {c.sqft.toLocaleString()} ft² · {c.days_on_market}d</div>
                          </div>
                          <div style={{ textAlign: "right" }}>
                            <div style={{ fontSize: 15, fontWeight: 800, fontFamily: "'Space Mono', monospace" }}>{fmt(c.price)}</div>
                            <span className="tag" style={{ background: c.status === "Active" ? "rgba(34,197,94,0.12)" : "rgba(148,163,184,0.1)", color: c.status === "Active" ? "#22C55E" : "#94A3B8", marginTop: 3 }}>{c.status}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* ── MARKET TAB ──────────────────────────────────────────────────── */}
        {tab === "market" && (
          <div style={{ animation: "fadeUp 0.5s ease" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16, marginBottom: 24 }}>
              {[
                { label: "Median Price", value: "$520K", change: "+8.3%", up: true },
                { label: "Active Listings", value: "1,284", change: "+3.1%", up: true },
                { label: "Days on Market", value: "18 days", change: "-22%", up: false },
                { label: "Price/sqft", value: "$285", change: "+6.7%", up: true },
              ].map((s, i) => (
                <div key={i} className="card">
                  <div style={{ fontSize: 11, color: "#8B9BB4", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 600, marginBottom: 10 }}>{s.label}</div>
                  <div style={{ fontSize: 28, fontWeight: 900, letterSpacing: "-0.02em", fontFamily: "'Space Mono', monospace" }}>{s.value}</div>
                  <div style={{ fontSize: 12, color: s.up ? "#22C55E" : "#F87171", marginTop: 6, fontWeight: 700 }}>{s.change} YoY</div>
                </div>
              ))}
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 24 }}>
              <div className="card">
                <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#8B9BB4", marginBottom: 24 }}>Average Price Trend — 2024</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={marketData}>
                    <defs>
                      <linearGradient id="priceGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#4F7FFF" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#4F7FFF" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="month" stroke="#8B9BB4" tick={{ fontSize: 12 }} />
                    <YAxis stroke="#8B9BB4" tick={{ fontSize: 12 }} tickFormatter={v => `$${v/1000}K`} />
                    <Tooltip contentStyle={{ background: "#0D1521", border: "1px solid rgba(79,127,255,0.3)", borderRadius: 10 }} formatter={v => [`$${v.toLocaleString()}`, "Avg Price"]} />
                    <Area type="monotone" dataKey="price" stroke="#4F7FFF" strokeWidth={2.5} fill="url(#priceGrad)" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              <div className="card">
                <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#8B9BB4", marginBottom: 24 }}>Neighborhoods</h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                  {[
                    { name: "Waterfront", price: "$1.2M", change: "+15.1%", score: 98 },
                    { name: "Downtown", price: "$750K", change: "+12.5%", score: 95 },
                    { name: "West Hills", price: "$620K", change: "+9.8%", score: 88 },
                    { name: "Suburbs North", price: "$520K", change: "+8.2%", score: 82 },
                    { name: "Old Town", price: "$430K", change: "+5.3%", score: 78 },
                    { name: "East Side", price: "$380K", change: "+6.7%", score: 74 },
                  ].map((n, i) => (
                    <div key={i} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "10px 14px", background: "#080E1A", borderRadius: 10 }}>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600 }}>{n.name}</div>
                        <div style={{ fontSize: 11, color: "#22C55E", marginTop: 2, fontWeight: 700 }}>{n.change}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: 14, fontWeight: 800, fontFamily: "'Space Mono', monospace" }}>{n.price}</div>
                        <div style={{ fontSize: 11, color: "#8B9BB4" }}>Score: {n.score}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── MODEL TAB ──────────────────────────────────────────────────── */}
        {tab === "about" && (
          <div style={{ animation: "fadeUp 0.5s ease", display: "grid", gridTemplateColumns: "1fr 1fr", gap: 24 }}>
            <div className="card">
              <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 24, letterSpacing: "-0.02em" }}>Model Performance</h2>
              {[
                { label: "R² Score", value: "0.887", bar: 88.7, color: "#4F7FFF" },
                { label: "Prediction Accuracy", value: "94.2%", bar: 94.2, color: "#22C55E" },
                { label: "Training Samples", value: "21,613", bar: 100, color: "#7B5EFF" },
                { label: "Features Used", value: "18 + 8 engineered", bar: 65, color: "#F59E0B" },
              ].map((m, i) => (
                <div key={i} style={{ marginBottom: 20 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <span style={{ fontSize: 13, color: "#8B9BB4", textTransform: "uppercase", letterSpacing: "0.07em", fontWeight: 600 }}>{m.label}</span>
                    <span style={{ fontSize: 14, fontWeight: 800, fontFamily: "'Space Mono', monospace", color: m.color }}>{m.value}</span>
                  </div>
                  <div style={{ height: 6, background: "#1E2A3E", borderRadius: 3 }}>
                    <div style={{ width: `${m.bar}%`, height: "100%", background: m.color, borderRadius: 3 }} />
                  </div>
                </div>
              ))}

              <div style={{ marginTop: 24, padding: "16px", background: "#080E1A", borderRadius: 12, border: "1px solid rgba(79,127,255,0.15)" }}>
                <div style={{ fontSize: 11, color: "#8B9BB4", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 12 }}>Error Metrics</div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                  {[["MAE", "$32,450"], ["RMSE", "$48,250"], ["MAPE", "7.8%"], ["Std Dev", "$51,200"]].map(([k, v]) => (
                    <div key={k}>
                      <div style={{ fontSize: 11, color: "#8B9BB4" }}>{k}</div>
                      <div style={{ fontSize: 16, fontWeight: 800, fontFamily: "'Space Mono', monospace", color: "#E8F0FF", marginTop: 2 }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              <div className="card">
                <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#8B9BB4", marginBottom: 20 }}>Technology Stack</h3>
                {[
                  { name: "XGBoost", role: "Primary prediction model", icon: "🚀" },
                  { name: "LightGBM", role: "Ensemble member", icon: "⚡" },
                  { name: "Random Forest", role: "Ensemble member", icon: "🌲" },
                  { name: "Scikit-learn", role: "Preprocessing & evaluation", icon: "🔬" },
                  { name: "FastAPI", role: "REST API backend", icon: "🔌" },
                  { name: "React + Recharts", role: "Frontend dashboard", icon: "📊" },
                ].map((t, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 14, padding: "10px 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <div style={{ fontSize: 22 }}>{t.icon}</div>
                    <div>
                      <div style={{ fontSize: 14, fontWeight: 700 }}>{t.name}</div>
                      <div style={{ fontSize: 12, color: "#8B9BB4" }}>{t.role}</div>
                    </div>
                  </div>
                ))}
              </div>

              <div className="card">
                <h3 style={{ fontSize: 14, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#8B9BB4", marginBottom: 16 }}>Engineered Features</h3>
                <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                  {["house_age", "was_renovated", "years_since_reno", "basement_ratio", "lot_ratio", "neighbor_ratio", "bath_per_bed", "total_rooms", "luxury_score", "premium_features"].map(f => (
                    <span key={f} className="tag" style={{ background: "rgba(79,127,255,0.1)", color: "#4F7FFF", fontFamily: "'Space Mono', monospace", fontSize: 10 }}>{f}</span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
