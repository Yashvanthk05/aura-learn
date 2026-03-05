import { useCallback, useState } from "react";
import { Search } from "lucide-react";
import { useApp } from "../../store/AppContext";
import * as api from "../../api/client";

export default function SearchFeature() {
  const { state, dispatch } = useApp();
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("hybrid");

  const run = useCallback(async () => {
    if (!query.trim()) return;
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      const res = await api.searchDocument(state.activeDocumentId, query, {
        method,
      });
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [state.activeDocumentId, query, method, dispatch]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") run();
  };

  return (
    <div className='space-y-4'>
      <div
        className='flex gap-1 p-1 rounded-lg'
        style={{ background: "var(--bg-elevated)" }}
      >
        {["hybrid", "faiss", "bm25"].map((m) => (
          <button
            key={m}
            onClick={() => setMethod(m)}
            className='flex-1 text-xs py-1.5 rounded-md transition-colors uppercase'
            style={{
              fontSize: 10,
              background: method === m ? "var(--accent-muted)" : "transparent",
              color: method === m ? "var(--accent)" : "var(--fg-tertiary)",
            }}
          >
            {m}
          </button>
        ))}
      </div>

      <div
        className='flex items-center gap-2 rounded-lg px-3 py-2'
        style={{
          background: "var(--bg-elevated)",
          border: "1px solid var(--border)",
        }}
      >
        <Search size={14} style={{ color: "var(--fg-tertiary)" }} />
        <input
          type='text'
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder='Search document...'
          className='flex-1 bg-transparent outline-none text-xs'
          style={{ color: "var(--fg-primary)" }}
        />
      </div>

      <button
        onClick={run}
        disabled={state.isFeatureLoading || !query.trim()}
        className='w-full py-2 rounded-lg text-xs font-medium transition-colors'
        style={{
          background: "var(--accent)",
          color: "#fff",
          opacity: state.isFeatureLoading || !query.trim() ? 0.5 : 1,
        }}
      >
        {state.isFeatureLoading ? "Searching…" : "Search"}
      </button>

      {state.featureResult?.results && (
        <div className='space-y-2'>
          <p
            className='text-xs'
            style={{ color: "var(--fg-muted)", fontSize: 10 }}
          >
            {state.featureResult.total_results} results via{" "}
            {state.featureResult.search_method}
          </p>
          {state.featureResult.results.map((r, i) => (
            <div
              key={i}
              className='rounded-lg p-3'
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
              }}
            >
              <div className='flex items-center justify-between mb-1'>
                <span
                  className='text-xs font-medium'
                  style={{ color: "var(--fg-primary)", fontSize: 11 }}
                >
                  {r.topic}
                </span>
                <span
                  className='text-xs px-1.5 py-0.5 rounded'
                  style={{
                    background: "var(--accent-muted)",
                    color: "var(--accent)",
                    fontSize: 9,
                  }}
                >
                  p.{r.page} · {(r.score * 100).toFixed(0)}%
                </span>
              </div>
              <p
                className='text-xs leading-relaxed'
                style={{ color: "var(--fg-secondary)" }}
              >
                {r.text?.slice(0, 200)}
                {r.text?.length > 200 ? "…" : ""}
              </p>
              {r.score_breakdown &&
                Object.keys(r.score_breakdown).length > 0 && (
                  <div className='flex gap-1.5 mt-2 flex-wrap'>
                    {Object.entries(r.score_breakdown).map(([k, v]) => (
                      <span
                        key={k}
                        className='text-xs px-1.5 py-0.5 rounded'
                        style={{
                          background: "var(--bg-overlay)",
                          color: "var(--fg-muted)",
                          fontSize: 8,
                          fontFamily: "var(--font-mono)",
                        }}
                      >
                        {k}: {typeof v === "number" ? v.toFixed(3) : v}
                      </span>
                    ))}
                  </div>
                )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
