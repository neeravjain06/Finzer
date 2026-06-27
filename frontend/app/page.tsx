"use client";

import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// --- Types ---
interface Signal {
  symbol: string;
  signal: string;
  confidence: number;
  bullish_prob: number;
  bearish_prob: number;
}

interface PriceBar {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface NewsItem {
  headline: string;
  source: string;
  published_at: string;
  sentiment_score: number;
  url: string;
}

// --- Signal Card ---
function SignalCard({ symbol }: { symbol: string }) {
  const [signal, setSignal] = useState<Signal | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/signals/${symbol}`)
      .then((r) => r.json())
      .then((data) => { setSignal(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [symbol]);

  if (loading) return (
    <div className="glass rounded-xl p-6 animate-pulse">
      <div className="h-4 bg-white/10 rounded w-1/2 mb-4"></div>
      <div className="h-8 bg-white/10 rounded w-3/4 mb-4"></div>
      <div className="h-2 bg-white/10 rounded w-full"></div>
    </div>
  );

  const isBullish = signal?.signal === "bullish";

  return (
    <div className="glass rounded-xl p-6 relative overflow-hidden group cursor-pointer">
      <div className={`absolute -right-10 -top-10 w-32 h-32 rounded-full blur-2xl transition-all ${isBullish ? 'bg-orange-500/10 group-hover:bg-orange-500/20' : 'bg-white/5 group-hover:bg-white/10'}`}></div>
      <div className="flex justify-between items-start mb-4 relative z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-white/5 flex items-center justify-center border border-white/10">
            <span className="material-symbols-outlined text-white">
              {symbol === "AAPL" ? "apps" : "currency_bitcoin"}
            </span>
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">{symbol}</h3>
            <p className="text-xs text-white/50">{symbol === "AAPL" ? "Apple Inc." : "Bitcoin"}</p>
          </div>
        </div>
        <div className={`px-2 py-1 rounded flex items-center gap-1 text-xs font-bold border ${isBullish ? 'bg-orange-500/10 border-orange-500/30 text-orange-400' : 'bg-white/10 border-white/20 text-white/70'}`}>
          <span className="material-symbols-outlined text-sm">
            {isBullish ? "trending_up" : "trending_down"}
          </span>
          {signal?.signal?.toUpperCase() ?? "UNKNOWN"}
        </div>
      </div>
      <div className="relative z-10">
        <div className="flex justify-between items-end mb-1">
          <span className="text-xs text-white/50">Model Confidence</span>
          <span className={`text-sm font-semibold ${isBullish ? 'text-orange-400' : 'text-white/70'}`}>
            {((signal?.confidence ?? 0) * 100).toFixed(1)}%
          </span>
        </div>
        <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full ${isBullish ? 'bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]' : 'bg-white/40'}`}
            style={{ width: `${(signal?.confidence ?? 0) * 100}%` }}
          ></div>
        </div>
        <div className="flex justify-between mt-2 text-xs text-white/40">
          <span>Bull: {((signal?.bullish_prob ?? 0) * 100).toFixed(1)}%</span>
          <span>Bear: {((signal?.bearish_prob ?? 0) * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

// --- Price Chart ---
function PriceChart({ symbol }: { symbol: string }) {
  const [bars, setBars] = useState<PriceBar[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/prices/${symbol}?limit=30`)
      .then((r) => r.json())
      .then((data) => { setBars(data.reverse()); setLoading(false); })
      .catch(() => setLoading(false));
  }, [symbol]);

  const latest = bars[bars.length - 1];
  const prices = bars.map((b) => b.close);
  const minPrice = Math.min(...prices);
  const maxPrice = Math.max(...prices);
  const range = maxPrice - minPrice || 1;

  if (loading) return (
    <div className="glass rounded-xl p-6 h-[400px] animate-pulse">
      <div className="h-full bg-white/5 rounded"></div>
    </div>
  );

  return (
    <div className="glass rounded-xl p-6 flex flex-col min-h-[400px]">
      <div className="flex items-end justify-between pb-4 border-b border-white/10 mb-4">
        <div>
          <p className="text-xs text-white/50 uppercase tracking-wider mb-1">{symbol} — Last 30 Days</p>
          <p className="text-2xl font-bold text-white">${latest?.close.toFixed(2)}</p>
        </div>
      </div>
      <div className="flex-1 flex items-end justify-between gap-1 pb-8 relative">
        {bars.slice(-20).map((bar, i) => {
          const prev = bars[i - 1];
          const isBull = prev ? bar.close >= prev.close : true;
          const height = ((bar.close - minPrice) / range) * 100;
          return (
            <div key={i} className="flex-1 flex flex-col items-center justify-end h-48 group">
              <div
                className={`w-full rounded-sm transition-all ${isBull ? 'bg-orange-500 group-hover:shadow-[0_0_8px_rgba(249,115,22,0.6)]' : 'bg-white/30'}`}
                style={{ height: `${Math.max(height, 4)}%` }}
                title={`Close: $${bar.close.toFixed(2)}`}
              ></div>
            </div>
          );
        })}
      </div>
      {latest && (
        <div className="flex justify-center gap-3 border-t border-white/10 pt-4 flex-wrap">
          {[
            { label: "O", value: latest.open.toFixed(2) },
            { label: "H", value: latest.high.toFixed(2), highlight: true },
            { label: "L", value: latest.low.toFixed(2) },
            { label: "C", value: latest.close.toFixed(2) },
          ].map(({ label, value, highlight }) => (
            <div key={label} className="flex items-center gap-1.5 bg-white/5 px-3 py-1.5 rounded-full border border-white/10">
              <span className="text-xs text-white/50">{label}:</span>
              <span className={`text-xs font-mono ${highlight ? 'text-orange-400' : 'text-white'}`}>${value}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --- News Feed ---
function NewsFeed({ symbol }: { symbol: string }) {
  const [news, setNews] = useState<NewsItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/news/${symbol}?limit=10`)
      .then((r) => r.json())
      .then((data) => { setNews(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [symbol]);

  const scoreColor = (score: number) => {
    if (score > 0.3) return "bg-orange-500/10 border-orange-500/30 text-orange-400";
    if (score < -0.3) return "bg-red-500/10 border-red-500/30 text-red-400";
    return "bg-white/10 border-white/20 text-white/50";
  };

  if (loading) return (
    <div className="glass rounded-xl p-4 h-[400px] animate-pulse space-y-3">
      {[...Array(5)].map((_, i) => (
        <div key={i} className="h-16 bg-white/5 rounded"></div>
      ))}
    </div>
  );

  return (
    <div className="glass rounded-xl overflow-hidden">
      <div className="p-4 border-b border-white/10">
        <h3 className="text-xs font-semibold text-white/50 uppercase tracking-wider">News Sentiment</h3>
      </div>
      <div className="divide-y divide-white/5 max-h-[400px] overflow-y-auto">
        {news.map((item, i) => (
          <a key={i} href={item.url} target="_blank" rel="noopener noreferrer"
            className="flex gap-3 items-start p-3 hover:bg-white/5 transition-colors group">
            <div className="flex-1 min-w-0">
              <p className="text-xs text-white line-clamp-2 leading-snug group-hover:text-orange-400 transition-colors">{item.headline}</p>
              <p className="text-[10px] text-white/40 mt-1 uppercase tracking-wide">
                {item.source} • {new Date(item.published_at).toLocaleDateString()}
              </p>
            </div>
            <div className={`px-2 py-0.5 rounded border text-[10px] font-bold shrink-0 ${scoreColor(item.sentiment_score)}`}>
              {item.sentiment_score > 0 ? "+" : ""}{item.sentiment_score.toFixed(2)}
            </div>
          </a>
        ))}
      </div>
    </div>
  );
}

// --- Navbar ---
function Navbar({ apiOnline, activeSymbol, setActiveSymbol }: {
  apiOnline: boolean;
  activeSymbol: string;
  setActiveSymbol: (s: string) => void;
}) {
  return (
    <nav className="fixed top-0 w-full z-50 bg-[#131313]/80 backdrop-blur-xl border-b border-white/10 shadow-[0_0_15px_rgba(249,115,22,0.1)]">
      <div className="max-w-[1440px] mx-auto px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-6">
          <span className="text-xl font-bold text-orange-500">Finzer</span>
          <div className="flex items-center gap-2 border-l border-white/10 pl-4">
            {["AAPL", "BTC-USD"].map((sym) => (
              <button key={sym} onClick={() => setActiveSymbol(sym)}
                className={`px-3 py-1 rounded-full text-xs font-semibold transition-all ${activeSymbol === sym ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50 shadow-[0_0_10px_rgba(249,115,22,0.3)]' : 'text-white/60 hover:text-white'}`}>
                {sym}
              </button>
            ))}
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${apiOnline ? 'bg-orange-500 shadow-[0_0_8px_rgba(249,115,22,0.6)]' : 'bg-red-500'}`}></span>
            <span className="text-xs text-white/50">{apiOnline ? "API Online" : "API Offline"}</span>
          </div>
        </div>
      </div>
    </nav>
  );
}

// --- Main Dashboard ---
export default function Dashboard() {
  const [activeSymbol, setActiveSymbol] = useState("AAPL");
  const [apiOnline, setApiOnline] = useState(false);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState("");

  useEffect(() => {
    fetch(`${API_BASE}/`)
      .then((r) => r.ok && setApiOnline(true))
      .catch(() => setApiOnline(false));
  }, []);

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg("");
    try {
      const r = await fetch(`${API_BASE}/train/${activeSymbol}`, { method: "POST" });
      const data = await r.json();
      setRetrainMsg(data.message ?? "Retrained successfully");
    } catch {
      setRetrainMsg("Retraining failed");
    } finally {
      setRetraining(false);
    }
  };

  return (
    <>
      <Navbar apiOnline={apiOnline} activeSymbol={activeSymbol} setActiveSymbol={setActiveSymbol} />
      <main className="pt-24 pb-16 px-6 max-w-[1440px] mx-auto">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white">Market Intelligence</h1>
          <p className="text-white/50 mt-1 text-sm">Real-time signal processing and sentiment analysis.</p>
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <div className="lg:col-span-3 flex flex-col gap-4">
            <p className="text-xs font-semibold text-white/40 uppercase tracking-wider">Active Signals</p>
            <SignalCard symbol="AAPL" />
            <SignalCard symbol="BTC-USD" />
            <div className="glass rounded-xl p-4">
              <p className="text-xs text-white/40 uppercase tracking-wider mb-3">Model Controls</p>
              <button onClick={handleRetrain} disabled={retraining}
                className="w-full py-2 rounded-lg bg-gradient-to-r from-orange-500 to-orange-600 text-white text-sm font-semibold hover:shadow-[0_0_15px_rgba(249,115,22,0.4)] transition-all disabled:opacity-50">
                {retraining ? "Retraining..." : `Retrain ${activeSymbol}`}
              </button>
              {retrainMsg && <p className="text-xs text-orange-400 mt-2">{retrainMsg}</p>}
            </div>
          </div>
          <div className="lg:col-span-6 flex flex-col gap-4">
            <p className="text-xs font-semibold text-white/40 uppercase tracking-wider">Price Action</p>
            <PriceChart symbol={activeSymbol} />
          </div>
          <div className="lg:col-span-3 flex flex-col gap-4">
            <p className="text-xs font-semibold text-white/40 uppercase tracking-wider">News & Sentiment</p>
            <NewsFeed symbol={activeSymbol} />
          </div>
        </div>
      </main>
    </>
  );
}