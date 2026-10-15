import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Free-tier hosts (Railway/Qdrant) spin down when idle and can take up to
// ~1-2 min to wake on the next request — poll /health at startup so the UI
// can say so instead of just looking broken while the first request hangs.
const HEALTH_CHECK_TIMEOUT_MS = 3000;
const HEALTH_CHECK_MAX_ATTEMPTS = 6;
const HEALTH_CHECK_RETRY_MS = 10000;

const SAMPLE_QUERIES = [
    "What was TCS revenue from operations in Q3FY25?",
    "What was the standalone Profit After Tax of Reliance Industries in FY25?",
    "What dividend did TCS declare in Q3FY25?",
    "What was Infosys consolidated revenue from operations in Q1FY27?",
];

export default function App() {
    const [query, setQuery] = useState("");
    const [company, setCompany] = useState("");
    const [period, setPeriod] = useState("");
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [expandedCitations, setExpandedCitations] = useState({});
    const [companies, setCompanies] = useState([]);
    const [backendStatus, setBackendStatus] = useState("checking"); // checking | online | waking | offline

    useEffect(() => {
        axios
            .get(`${API_BASE}/companies`)
            .then((res) => setCompanies(res.data.companies || []))
            .catch(() => setCompanies([]));
    }, []);

    useEffect(() => {
        let cancelled = false;
        let attempt = 0;

        const checkHealth = async () => {
            attempt += 1;
            try {
                await axios.get(`${API_BASE}/health`, { timeout: HEALTH_CHECK_TIMEOUT_MS });
                if (!cancelled) setBackendStatus("online");
            } catch {
                if (cancelled) return;
                if (attempt >= HEALTH_CHECK_MAX_ATTEMPTS) {
                    setBackendStatus("offline");
                } else {
                    setBackendStatus("waking");
                    setTimeout(checkHealth, HEALTH_CHECK_RETRY_MS);
                }
            }
        };

        checkHealth();
        return () => {
            cancelled = true;
        };
    }, []);

    const handleSubmit = async (q = query) => {
        if (!q.trim()) return;
        setLoading(true);
        setResult(null);
        setError(null);
        setExpandedCitations({});

        try {
            const payload = { query: q.trim(), top_k: 5 };
            if (company) payload.company = company;
            if (period) payload.period = period;

            const res = await axios.post(`${API_BASE}/query`, payload);
            setResult(res.data);
            setBackendStatus("online");
        } catch (err) {
            setError(
                err.response?.data?.detail ||
                "Could not reach the API. Make sure the backend is running."
            );
        } finally {
            setLoading(false);
        }
    };

    const toggleCitation = (idx) => {
        setExpandedCitations((prev) => ({ ...prev, [idx]: !prev[idx] }));
    };

    const useSample = (q) => {
        setQuery(q);
        handleSubmit(q);
    };

    return (
        <div className="app">
            <header className="header">
                <div className="header-inner">
                    <div className="logo">
                        <span className="logo-mark">₹</span>
                        <div>
                            <h1 className="logo-title">FinFiling RAG</h1>
                            <p className="logo-sub">SEBI / BSE Filing Analyzer</p>
                        </div>
                    </div>
                    <div className={`header-badge status-${backendStatus}`}>
                        <span className="badge-dot" />
                        {backendStatus === "online" && "Live"}
                        {backendStatus === "checking" && "Connecting…"}
                        {backendStatus === "waking" && "Waking up…"}
                        {backendStatus === "offline" && "Offline"}
                    </div>
                </div>
            </header>

            <main className="main">
                <section className="search-section">
                    <p className="search-label">Ask anything about indexed filings</p>
                    <div className="search-box">
                        <textarea
                            className="search-input"
                            placeholder="e.g. What was TCS net profit in Q3FY25?"
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            onKeyDown={(e) => {
                                if (e.key === "Enter" && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSubmit();
                                }
                            }}
                            rows={2}
                        />
                        <button
                            className="search-btn"
                            onClick={() => handleSubmit()}
                            disabled={loading || !query.trim()}
                        >
                            {loading ? <span className="spinner" /> : "Ask"}
                        </button>
                    </div>

                    <div className="filters">
                        <select className="filter-select" value={company} onChange={(e) => setCompany(e.target.value)}>
                            <option value="">Auto-detect from question</option>
                            {companies.map((name) => (
                                <option key={name} value={name}>{name}</option>
                            ))}
                        </select>
                        <input
                            className="filter-input"
                            placeholder="Period (auto) e.g. Q3FY25"
                            value={period}
                            onChange={(e) => setPeriod(e.target.value)}
                        />
                    </div>

                    <div className="samples">
                        <span className="samples-label">Try:</span>
                        {SAMPLE_QUERIES.map((q) => (
                            <button key={q} className="sample-chip" onClick={() => useSample(q)}>{q}</button>
                        ))}
                    </div>
                </section>

                {(backendStatus === "waking" || backendStatus === "offline") && !loading && (
                    <div className="wake-banner">
                        {backendStatus === "waking"
                            ? "The backend is waking up from idle (free-tier hosting) — the first response can take up to a minute."
                            : "Can't reach the backend right now. It may still be waking up — try again in a moment."}
                    </div>
                )}

                {error && (
                    <div className="error-box"><span className="error-icon">⚠</span> {error}</div>
                )}

                {loading && (
                    <div className="loading-box">
                        <div className="loading-bar" />
                        <p>Retrieving from filings and generating answer…</p>
                        {backendStatus !== "online" && (
                            <p className="loading-hint">
                                First request after idle can take up to a minute while the server wakes up.
                            </p>
                        )}
                    </div>
                )}

                {result && !loading && (
                    <section className="result-section">
                        <div className="result-meta">
                            <span className={`grounding-badge ${result.is_grounded ? "grounded" : "ungrounded"}`}>
                                {result.is_grounded ? "✓ Grounded" : "⚠ Unverified"}
                            </span>
                            <span className="meta-pill">Confidence {(result.grounding_confidence * 100).toFixed(0)}%</span>
                            <span className="meta-pill">{result.chunks_used} chunks</span>
                            <span className="meta-pill latency">{result.latency_ms.toFixed(0)} ms</span>
                            {result.path === "agent" && (
                                <span className="meta-pill" title={(result.tool_calls || []).map((t) => t.tool).join(" → ")}>
                                    Computed with tools · {(result.tool_calls || []).length} calls
                                </span>
                            )}
                            {result.filters_applied?.filters?.length > 0 && (
                                <span className="meta-pill" title="Company and period used to filter the search">
                                    Searched: {result.filters_applied.filters
                                        .map((f) => [f.company, f.period].filter(Boolean).join(" · "))
                                        .join(" + ")}
                                </span>
                            )}
                        </div>

                        {result.warning && <div className="warning-box">{result.warning}</div>}

                        <div className="answer-card">
                            <div className="answer-label">Answer</div>
                            <div className="answer-text">{result.answer}</div>
                        </div>

                        {result.citations && result.citations.length > 0 && (
                            <div className="citations">
                                <div className="citations-header">Sources ({result.citations.length})</div>
                                {result.citations.map((c, idx) => (
                                    <div key={idx} className="citation-card">
                                        <div className="citation-summary" onClick={() => toggleCitation(idx)}>
                                            <div className="citation-info">
                                                <span className="citation-company">{c.company}</span>
                                                <span className="citation-period">{c.period}</span>
                                                <span className="citation-type">{c.doc_type?.replace(/_/g, " ")}</span>
                                                <span className="citation-page">p. {c.page_number}</span>
                                            </div>
                                            <span className="citation-toggle">{expandedCitations[idx] ? "▲" : "▼"}</span>
                                        </div>
                                        {expandedCitations[idx] && (
                                            <div className="citation-excerpt">
                                                <div className="excerpt-label">Excerpt</div>
                                                <p>{c.excerpt}</p>
                                                <div className="excerpt-file">{c.source_file}</div>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                        )}
                    </section>
                )}

                {!result && !loading && !error && (
                    <div className="empty-state">
                        <div className="empty-icon">📄</div>
                        <p>
                            {companies.length > 0
                                ? `Ask a question about ${companies.join(" or ")} filings to get started.`
                                : "Ask a question about an indexed company's filings to get started."}
                        </p>
                    </div>
                )}
            </main>

            <footer className="footer">
                Answers grounded in official SEBI/BSE filings · Not financial advice
            </footer>
        </div>
    );
}
