import { useCallback, useState } from "react";
import {
  FileText,
  AudioLines,
  Search,
  BrainCircuit,
  Loader2,
  Play,
  Pause,
  ChevronDown,
} from "lucide-react";
import { useApp } from "../store/AppContext";
import * as api from "../api/client";

const TABS = [
  { id: "summarize", label: "Summarize", icon: FileText },
  { id: "audio", label: "Audio", icon: AudioLines },
  { id: "search", label: "Search", icon: Search },
  { id: "xai", label: "Explain", icon: BrainCircuit },
];

function SummarizeFeature() {
  const { state, dispatch } = useApp();
  const [type, setType] = useState("extractive");
  const [sentences, setSentences] = useState(5);
  const [maxLength, setMaxLength] = useState(150);
  const [minLength, setMinLength] = useState(40);

  const run = useCallback(async () => {
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      const res = await api.summarize(state.activeDocumentId, {
        type,
        numSentences: type === "extractive" ? sentences : undefined,
        maxLength: type === "abstractive" ? maxLength : undefined,
        minLength: type === "abstractive" ? minLength : undefined,
      });
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [state.activeDocumentId, type, sentences, maxLength, minLength, dispatch]);

  return (
    <div className="space-y-4">
      <div className="flex gap-1 p-1 rounded-lg" style={{ background: "var(--bg-elevated)" }}>
        {["extractive", "abstractive"].map((t) => (
          <button key={t} onClick={() => setType(t)}
            className="flex-1 text-xs py-1.5 rounded-md transition-colors capitalize"
            style={{
              background: type === t ? "var(--accent-muted)" : "transparent",
              color: type === t ? "var(--accent)" : "var(--fg-tertiary)",
            }}>{t}</button>
        ))}
      </div>

      {type === "extractive" && (
        <div>
          <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
            Sentences: {sentences}
          </label>
          <input type="range" min={1} max={15} value={sentences}
            onChange={(e) => setSentences(Number(e.target.value))} className="w-full slider-styled" />
        </div>
      )}

      {type === "abstractive" && (
        <>
          <div>
            <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
              Max length: {maxLength}
            </label>
            <input type="range" min={50} max={500} step={10} value={maxLength}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMaxLength(v);
                if (minLength >= v) setMinLength(Math.max(20, v - 20));
              }} className="w-full slider-styled" />
          </div>
          <div>
            <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
              Min length: {minLength}
            </label>
            <input type="range" min={20} max={300} step={10} value={minLength}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMinLength(v);
                if (maxLength <= v) setMaxLength(v + 20);
              }} className="w-full slider-styled" />
          </div>
        </>
      )}

      <button onClick={run} disabled={state.isFeatureLoading}
        className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
        style={{ background: "var(--accent)", color: "#fff", opacity: state.isFeatureLoading ? 0.5 : 1 }}>
        {state.isFeatureLoading ? "Generating…" : "Summarize"}
      </button>

      {state.featureResult?.summary && (
        <>
          <div className="flex items-center gap-2 flex-wrap" style={{ fontSize: 10, color: "var(--fg-muted)" }}>
            <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--bg-elevated)" }}>
              {state.featureResult.summarization_type}
            </span>
            <span>{state.featureResult.num_chunks_processed} chunks processed</span>
          </div>
          <div className="rounded-lg p-3 text-sm leading-relaxed"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--fg-primary)" }}>
            {state.featureResult.summary}
          </div>
          {state.featureResult.metadata && Object.keys(state.featureResult.metadata).length > 0 && (
            <div className="rounded-lg p-2 space-y-1" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
              {Object.entries(state.featureResult.metadata).map(([k, v]) => (
                <div key={k} className="flex justify-between text-xs" style={{ fontSize: 10 }}>
                  <span style={{ color: "var(--fg-muted)" }}>{k}</span>
                  <span style={{ color: "var(--fg-secondary)" }}>{typeof v === "number" ? v.toFixed(2) : String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}

function AudioFeature() {
  const { state, dispatch } = useApp();
  const [playing, setPlaying] = useState(false);
  const [audioEl, setAudioEl] = useState(null);
  const [maxLength, setMaxLength] = useState(150);
  const [minLength, setMinLength] = useState(40);

  const run = useCallback(async () => {
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      const res = await api.summarizeAndAudio(state.activeDocumentId, {
        type: "abstractive",
        maxLength,
        minLength,
      });
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [state.activeDocumentId, maxLength, minLength, dispatch]);

  const togglePlay = useCallback(() => {
    if (!audioEl) return;
    if (playing) {
      audioEl.pause();
    } else {
      audioEl.play();
    }
    setPlaying(!playing);
  }, [audioEl, playing]);

  return (
    <div className="space-y-4">
      <div>
        <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
          Max length: {maxLength}
        </label>
        <input type="range" min={50} max={500} step={10} value={maxLength}
          onChange={(e) => {
            const v = Number(e.target.value);
            setMaxLength(v);
            if (minLength >= v) setMinLength(Math.max(20, v - 20));
          }} className="w-full slider-styled" />
      </div>
      <div>
        <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
          Min length: {minLength}
        </label>
        <input type="range" min={20} max={300} step={10} value={minLength}
          onChange={(e) => {
            const v = Number(e.target.value);
            setMinLength(v);
            if (maxLength <= v) setMaxLength(v + 20);
          }} className="w-full slider-styled" />
      </div>

      <button onClick={run} disabled={state.isFeatureLoading}
        className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
        style={{ background: "var(--accent)", color: "#fff", opacity: state.isFeatureLoading ? 0.5 : 1 }}>
        {state.isFeatureLoading ? "Generating audio…" : "Generate Audio Summary"}
      </button>

      {state.featureResult?.summary && (
        <>
          <div className="flex items-center gap-2 flex-wrap" style={{ fontSize: 10, color: "var(--fg-muted)" }}>
            <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--bg-elevated)" }}>
              {state.featureResult.summarization_type}
            </span>
            <span>{state.featureResult.num_chunks_processed} chunks</span>
          </div>
          <div className="rounded-lg p-3 text-sm leading-relaxed"
            style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--fg-primary)" }}>
            {state.featureResult.summary}
          </div>
        </>
      )}

      {state.featureResult?.audio_url && (
        <div className="flex items-center gap-3 rounded-lg p-3"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
          <button onClick={togglePlay}
            className="w-8 h-8 rounded-full flex items-center justify-center shrink-0"
            style={{ background: "var(--accent)" }}>
            {playing ? <Pause size={14} color="#fff" /> : <Play size={14} color="#fff" className="ml-0.5" />}
          </button>
          <audio ref={(el) => setAudioEl(el)} src={state.featureResult.audio_url}
            onEnded={() => setPlaying(false)} className="hidden" />
          <span className="text-xs truncate" style={{ color: "var(--fg-secondary)" }}>
            {state.featureResult.audio_filename || "audio.wav"}
          </span>
        </div>
      )}
    </div>
  );
}

function SearchFeature() {
  const { state, dispatch } = useApp();
  const [query, setQuery] = useState("");
  const [method, setMethod] = useState("hybrid");

  const run = useCallback(async () => {
    if (!query.trim()) return;
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      const res = await api.searchDocument(state.activeDocumentId, query, { method });
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [state.activeDocumentId, query, method, dispatch]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter") run();
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-1 p-1 rounded-lg" style={{ background: "var(--bg-elevated)" }}>
        {["hybrid", "faiss", "bm25"].map((m) => (
          <button key={m} onClick={() => setMethod(m)}
            className="flex-1 text-xs py-1.5 rounded-md transition-colors uppercase"
            style={{
              fontSize: 10,
              background: method === m ? "var(--accent-muted)" : "transparent",
              color: method === m ? "var(--accent)" : "var(--fg-tertiary)",
            }}>{m}</button>
        ))}
      </div>

      <div className="flex items-center gap-2 rounded-lg px-3 py-2"
        style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
        <Search size={14} style={{ color: "var(--fg-tertiary)" }} />
        <input type="text" value={query} onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown} placeholder="Search document..."
          className="flex-1 bg-transparent outline-none text-xs" style={{ color: "var(--fg-primary)" }} />
      </div>

      <button onClick={run} disabled={state.isFeatureLoading || !query.trim()}
        className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
        style={{ background: "var(--accent)", color: "#fff", opacity: (state.isFeatureLoading || !query.trim()) ? 0.5 : 1 }}>
        {state.isFeatureLoading ? "Searching…" : "Search"}
      </button>

      {state.featureResult?.results && (
        <div className="space-y-2">
          <p className="text-xs" style={{ color: "var(--fg-muted)", fontSize: 10 }}>
            {state.featureResult.total_results} results via {state.featureResult.search_method}
          </p>
          {state.featureResult.results.map((r, i) => (
            <div key={i} className="rounded-lg p-3"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs font-medium" style={{ color: "var(--fg-primary)", fontSize: 11 }}>
                  {r.topic}
                </span>
                <span className="text-xs px-1.5 py-0.5 rounded"
                  style={{ background: "var(--accent-muted)", color: "var(--accent)", fontSize: 9 }}>
                  p.{r.page} · {(r.score * 100).toFixed(0)}%
                </span>
              </div>
              <p className="text-xs leading-relaxed" style={{ color: "var(--fg-secondary)" }}>
                {r.text?.slice(0, 200)}{r.text?.length > 200 ? "…" : ""}
              </p>
              {r.score_breakdown && Object.keys(r.score_breakdown).length > 0 && (
                <div className="flex gap-1.5 mt-2 flex-wrap">
                  {Object.entries(r.score_breakdown).map(([k, v]) => (
                    <span key={k} className="text-xs px-1.5 py-0.5 rounded"
                      style={{ background: "var(--bg-overlay)", color: "var(--fg-muted)", fontSize: 8, fontFamily: "var(--font-mono)" }}>
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

function XaiFeature() {
  const { state, dispatch } = useApp();
  const [mode, setMode] = useState("extractive");
  const [numSentences, setNumSentences] = useState(3);
  const [maxLength, setMaxLength] = useState(150);
  const [minLength, setMinLength] = useState(50);

  const run = useCallback(async () => {
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      let res;
      if (mode === "extractive") {
        res = await api.explainExtractive(state.activeDocumentId, { numSentences });
      } else {
        res = await api.explainAbstractive(state.activeDocumentId, { maxLength, minLength });
      }
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [state.activeDocumentId, mode, numSentences, maxLength, minLength, dispatch]);

  const r = state.featureResult;

  return (
    <div className="space-y-4">
      <div className="flex gap-1 p-1 rounded-lg" style={{ background: "var(--bg-elevated)" }}>
        {["extractive", "abstractive"].map((m) => (
          <button key={m} onClick={() => setMode(m)}
            className="flex-1 text-xs py-1.5 rounded-md transition-colors capitalize"
            style={{
              background: mode === m ? "var(--accent-muted)" : "transparent",
              color: mode === m ? "var(--accent)" : "var(--fg-tertiary)",
            }}>{m}</button>
        ))}
      </div>

      {mode === "extractive" && (
        <div>
          <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
            Sentences: {numSentences}
          </label>
          <input type="range" min={1} max={10} value={numSentences}
            onChange={(e) => setNumSentences(Number(e.target.value))} className="w-full slider-styled" />
        </div>
      )}

      {mode === "abstractive" && (
        <>
          <div>
            <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
              Max length: {maxLength}
            </label>
            <input type="range" min={50} max={500} step={10} value={maxLength}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMaxLength(v);
                if (minLength >= v) setMinLength(Math.max(20, v - 20));
              }} className="w-full slider-styled" />
          </div>
          <div>
            <label className="text-xs mb-1 block" style={{ color: "var(--fg-secondary)" }}>
              Min length: {minLength}
            </label>
            <input type="range" min={20} max={300} step={10} value={minLength}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMinLength(v);
                if (maxLength <= v) setMaxLength(v + 20);
              }} className="w-full slider-styled" />
          </div>
        </>
      )}

      <button onClick={run} disabled={state.isFeatureLoading}
        className="w-full py-2 rounded-lg text-xs font-medium transition-colors"
        style={{ background: "var(--accent)", color: "#fff", opacity: state.isFeatureLoading ? 0.5 : 1 }}>
        {state.isFeatureLoading ? "Analyzing…" : "Explain"}
      </button>

      {r?.xai_type === "extractive" && (
        <>
          <div className="grid grid-cols-2 gap-2">
            {[
              ["Input sentences", r.num_sentences_input],
              ["Selected", r.num_sentences_selected],
              ["Avg selected", `${(r.average_score_selected * 100).toFixed(1)}%`],
              ["Avg all", `${(r.average_score_all * 100).toFixed(1)}%`],
            ].map(([label, val]) => (
              <div key={label} className="rounded-md p-2" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
                <p className="text-xs" style={{ color: "var(--fg-muted)", fontSize: 9 }}>{label}</p>
                <p className="text-sm font-medium" style={{ color: "var(--fg-primary)" }}>{val}</p>
              </div>
            ))}
          </div>

          {r.score_distribution && Object.keys(r.score_distribution).length > 0 && (
            <div>
              <p className="text-xs mb-2" style={{ color: "var(--fg-muted)", fontSize: 10 }}>Score Distribution</p>
              <div className="space-y-1">
                {Object.entries(r.score_distribution).map(([range, count]) => (
                  <div key={range} className="flex items-center gap-2">
                    <span className="text-xs w-16 shrink-0" style={{ color: "var(--fg-tertiary)", fontSize: 9, fontFamily: "var(--font-mono)" }}>{range}</span>
                    <div className="flex-1 h-2 rounded-full" style={{ background: "var(--bg-elevated)" }}>
                      <div className="h-full rounded-full" style={{
                        width: `${Math.min(count / (r.num_sentences_input || 1) * 100, 100)}%`,
                        background: "var(--accent)", opacity: 0.7,
                      }} />
                    </div>
                    <span className="text-xs w-4 text-right" style={{ color: "var(--fg-muted)", fontSize: 9 }}>{count}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {r.summary && (
            <div className="rounded-lg p-3 text-sm leading-relaxed"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--fg-primary)" }}>
              {r.summary}
            </div>
          )}

          {r.sentences && (
            <div className="space-y-2">
              <p className="text-xs" style={{ color: "var(--fg-muted)", fontSize: 10 }}>Sentence Analysis</p>
              {r.sentences.map((s, i) => (
                <div key={i} className="rounded-lg p-2.5 space-y-1.5"
                  style={{
                    background: "var(--bg-elevated)",
                    border: `1px solid ${s.is_selected ? "var(--accent)" : "var(--border)"}`,
                    borderLeftWidth: 3,
                    borderLeftColor: s.is_selected ? "var(--accent)" : "var(--border)",
                  }}>
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="text-xs font-medium" style={{ color: s.is_selected ? "var(--accent)" : "var(--fg-secondary)", fontSize: 10 }}>
                      S{s.index + 1}
                    </span>
                    {s.is_selected && (
                      <span className="text-xs px-1 py-0.5 rounded" style={{ background: "var(--accent-muted)", color: "var(--accent)", fontSize: 8 }}>SELECTED</span>
                    )}
                    <span className="text-xs" style={{ color: "var(--fg-muted)", fontSize: 9 }}>{s.word_count} words</span>
                  </div>
                  <p className="text-xs leading-relaxed" style={{ color: "var(--fg-secondary)" }}>{s.text}</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-1.5 rounded-full" style={{ background: "var(--bg-overlay)" }}>
                      <div className="h-full rounded-full" style={{
                        width: `${Math.max(s.importance_score * 100, 2)}%`,
                        background: s.is_selected ? "var(--accent)" : "var(--fg-muted)",
                      }} />
                    </div>
                    <span className="text-xs shrink-0" style={{ color: s.is_selected ? "var(--accent)" : "var(--fg-tertiary)", fontSize: 9 }}>
                      {(s.importance_score * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex gap-3 flex-wrap" style={{ fontSize: 9, color: "var(--fg-muted)" }}>
                    <span>Sensitivity: {s.sensitivity?.toFixed(3) ?? "—"}</span>
                    {s.most_attended_sentence != null && <span>Attends → S{s.most_attended_sentence + 1}</span>}
                  </div>
                  {s.attention_to_others && s.attention_to_others.length > 0 && (
                    <div className="flex gap-0.5 flex-wrap">
                      {s.attention_to_others.map((a, j) => (
                        <span key={j} className="px-1 py-0.5 rounded text-xs"
                          style={{
                            fontSize: 8,
                            background: `rgba(78, 197, 137, ${Math.min(a, 1) * 0.4})`,
                            color: a > 0.3 ? "var(--accent)" : "var(--fg-muted)",
                          }}>
                          S{j + 1}:{(a * 100).toFixed(0)}%
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}

          {r.explanation_methods && (
            <div className="flex gap-1 flex-wrap">
              {r.explanation_methods.map((m) => (
                <span key={m} className="text-xs px-1.5 py-0.5 rounded"
                  style={{ background: "var(--bg-elevated)", color: "var(--fg-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}>{m}</span>
              ))}
            </div>
          )}
        </>
      )}

      {r?.xai_type === "abstractive" && (
        <>
          {r.summary && (
            <div className="rounded-lg p-3 text-sm leading-relaxed"
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--fg-primary)" }}>
              {r.summary}
            </div>
          )}

          <div className="flex gap-2 flex-wrap" style={{ fontSize: 10, color: "var(--fg-muted)" }}>
            <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--bg-elevated)" }}>
              {r.summary_word_count} tokens
            </span>
            <span className="px-1.5 py-0.5 rounded" style={{ background: "var(--bg-elevated)" }}>
              {(r.compression_ratio * 100).toFixed(1)}% compression
            </span>
          </div>

          {r.sentence_contributions && r.sentence_contributions.length > 0 && (
            <div>
              <p className="text-xs mb-2" style={{ color: "var(--fg-muted)", fontSize: 10 }}>Source Sentence Contributions</p>
              <div className="space-y-2">
                {r.sentence_contributions.map((sc, i) => {
                  const isMost = r.most_influential_sentence?.index === sc.index;
                  return (
                    <div key={i} className="rounded-lg p-2.5 space-y-1"
                      style={{
                        background: "var(--bg-elevated)",
                        border: `1px solid ${isMost ? "var(--accent)" : "var(--border)"}`,
                        borderLeftWidth: 3,
                        borderLeftColor: isMost ? "var(--accent)" : "var(--border)",
                      }}>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-medium" style={{ color: isMost ? "var(--accent)" : "var(--fg-secondary)", fontSize: 10 }}>
                          S{sc.index + 1}
                        </span>
                        {isMost && (
                          <span className="text-xs px-1 py-0.5 rounded" style={{ background: "var(--accent-muted)", color: "var(--accent)", fontSize: 8 }}>
                            MOST INFLUENTIAL
                          </span>
                        )}
                      </div>
                      <p className="text-xs leading-relaxed" style={{ color: "var(--fg-secondary)" }}>{sc.text}</p>
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 rounded-full" style={{ background: "var(--bg-overlay)" }}>
                          <div className="h-full rounded-full" style={{
                            width: `${Math.max((sc.normalized_contribution || sc.contribution_score) * 100, 2)}%`,
                            background: isMost ? "var(--accent)" : "var(--fg-muted)",
                          }} />
                        </div>
                        <span className="text-xs shrink-0" style={{ color: isMost ? "var(--accent)" : "var(--fg-tertiary)", fontSize: 9 }}>
                          {((sc.normalized_contribution || sc.contribution_score) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {r.token_confidence && (
            <div>
              <p className="text-xs mb-2" style={{ color: "var(--fg-muted)", fontSize: 10 }}>Token Confidence</p>
              <div className="flex flex-wrap gap-0.5">
                {r.token_confidence.map((t, i) => (
                  <span key={i} className="text-xs px-1 py-0.5 rounded cursor-default"
                    title={`${(t.confidence * 100).toFixed(1)}% confidence`}
                    style={{
                      fontSize: 10,
                      background: `rgba(78, 197, 137, ${Math.min(t.confidence, 1) * 0.3})`,
                      color: t.is_high_confidence ? "var(--accent)" : "var(--fg-tertiary)",
                      borderBottom: t.is_high_confidence ? "1px solid var(--accent)" : "1px solid transparent",
                    }}>
                    {t.token}
                  </span>
                ))}
              </div>
            </div>
          )}

          {r.original_text_sentences && r.original_text_sentences.length > 0 && (
            <details className="group">
              <summary className="text-xs cursor-pointer list-none flex items-center gap-1" style={{ color: "var(--fg-muted)", fontSize: 10 }}>
                <ChevronDown size={12} className="group-open:rotate-180 transition-transform" />
                Source text ({r.original_text_sentences.length} sentences)
              </summary>
              <div className="mt-2 rounded-lg p-2 space-y-1" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
                {r.original_text_sentences.map((s, i) => (
                  <p key={i} className="text-xs leading-relaxed" style={{ color: "var(--fg-secondary)", fontSize: 11 }}>
                    <span style={{ color: "var(--fg-muted)", fontFamily: "var(--font-mono)", fontSize: 9 }}>{i + 1}. </span>{s}
                  </p>
                ))}
              </div>
            </details>
          )}

          {r.explanation_methods && (
            <div className="flex gap-1 flex-wrap">
              {r.explanation_methods.map((m) => (
                <span key={m} className="text-xs px-1.5 py-0.5 rounded"
                  style={{ background: "var(--bg-elevated)", color: "var(--fg-muted)", fontSize: 9, fontFamily: "var(--font-mono)" }}>{m}</span>
              ))}
            </div>
          )}
        </>
      )}

      {r && !r.xai_type && r.summary && (
        <div className="rounded-lg p-3 text-sm leading-relaxed"
          style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)", color: "var(--fg-primary)" }}>
          {r.summary}
        </div>
      )}
    </div>
  );
}

export default function FeaturesPanel() {
  const { state, dispatch } = useApp();
  const [tab, setTab] = useState("summarize");

  const selectTab = (id) => {
    setTab(id);
    dispatch({ type: "SET_ACTIVE_FEATURE", payload: id });
  };

  if (!state.activeDocumentId) {
    return (
      <aside
        className="flex items-center justify-center h-full border-l"
        style={{
          width: 320,
          minWidth: 320,
          borderColor: "var(--border)",
          background: "var(--bg-surface)",
        }}
      >
        <p
          className="text-xs"
          style={{ color: "var(--fg-muted)" }}
        >
          Select a source to use features
        </p>
      </aside>
    );
  }

  const FeatureComponent = {
    summarize: SummarizeFeature,
    audio: AudioFeature,
    search: SearchFeature,
    xai: XaiFeature,
  }[tab];

  return (
    <aside
      className="flex flex-col h-full border-l"
      style={{
        width: 320,
        minWidth: 320,
        borderColor: "var(--border)",
        background: "var(--bg-surface)",
      }}
    >
      <div
        className="grid grid-cols-2 gap-1 p-2 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        {TABS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => selectTab(id)}
            className="flex items-center justify-center gap-2 py-2.5 rounded-lg text-xs transition-colors"
            style={{
              background: tab === id ? "var(--accent-muted)" : "transparent",
              color: tab === id ? "var(--accent)" : "var(--fg-tertiary)",
            }}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-4">
        {state.featureError && (
          <p
            className="text-xs mb-3 px-2 py-1.5 rounded-md"
            style={{
              color: "var(--error)",
              background: "rgba(248,113,113,0.08)",
            }}
          >
            {state.featureError}
          </p>
        )}

        {state.isFeatureLoading && (
          <div className="flex items-center justify-center py-8">
            <Loader2 size={20} className="animate-spin" style={{ color: "var(--accent)" }} />
          </div>
        )}

        <div style={{ display: state.isFeatureLoading ? 'none' : 'block' }}>
          <FeatureComponent />
        </div>
      </div>
    </aside>
  );
}
