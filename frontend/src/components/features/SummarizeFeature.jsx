import { useCallback, useState } from "react";
import { useApp } from "../../store/AppContext";
import * as api from "../../api/client";

export default function SummarizeFeature() {
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

  const formatMetadataValue = (value) => {
    if (Array.isArray(value)) return value.join(", ");
    if (typeof value === "number") return value.toFixed(2);
    return String(value);
  };

  return (
    <div className='space-y-4'>
      <div
        className='flex gap-1 p-1 rounded-lg'
        style={{ background: "var(--bg-elevated)" }}
      >
        {["extractive", "abstractive"].map((t) => (
          <button
            key={t}
            onClick={() => setType(t)}
            className='flex-1 text-xs py-1.5 rounded-md transition-colors capitalize'
            style={{
              background: type === t ? "var(--accent-muted)" : "transparent",
              color: type === t ? "var(--accent)" : "var(--fg-tertiary)",
            }}
          >
            {t}
          </button>
        ))}
      </div>

      {type === "extractive" && (
        <div>
          <label
            className='text-xs mb-1 block'
            style={{ color: "var(--fg-secondary)" }}
          >
            Sentences: {sentences}
          </label>
          <input
            type='range'
            min={1}
            max={15}
            value={sentences}
            onChange={(e) => setSentences(Number(e.target.value))}
            className='w-full slider-styled'
          />
        </div>
      )}

      {type === "abstractive" && (
        <>
          <div>
            <label
              className='text-xs mb-1 block'
              style={{ color: "var(--fg-secondary)" }}
            >
              Max length: {maxLength}
            </label>
            <input
              type='range'
              min={50}
              max={500}
              step={10}
              value={maxLength}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMaxLength(v);
                if (minLength >= v) setMinLength(Math.max(20, v - 20));
              }}
              className='w-full slider-styled'
            />
          </div>
          <div>
            <label
              className='text-xs mb-1 block'
              style={{ color: "var(--fg-secondary)" }}
            >
              Min length: {minLength}
            </label>
            <input
              type='range'
              min={20}
              max={300}
              step={10}
              value={minLength}
              onChange={(e) => {
                const v = Number(e.target.value);
                setMinLength(v);
                if (maxLength <= v) setMaxLength(v + 20);
              }}
              className='w-full slider-styled'
            />
          </div>
        </>
      )}

      <button
        onClick={run}
        disabled={state.isFeatureLoading}
        className='w-full py-2 rounded-lg text-xs font-medium transition-colors'
        style={{
          background: "var(--accent)",
          color: "#fff",
          opacity: state.isFeatureLoading ? 0.5 : 1,
        }}
      >
        {state.isFeatureLoading ? "Generating…" : "Summarize"}
      </button>

      {state.featureResult?.summary && (
        <>
          <div
            className='flex items-center gap-2 flex-wrap'
            style={{ fontSize: 10, color: "var(--fg-muted)" }}
          >
            <span
              className='px-1.5 py-0.5 rounded'
              style={{ background: "var(--bg-elevated)" }}
            >
              {state.featureResult.summarization_type}
            </span>
            <span>
              {state.featureResult.num_chunks_processed} chunks processed
            </span>
          </div>
          <div
            className='rounded-lg p-3 text-sm leading-relaxed'
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              color: "var(--fg-primary)",
            }}
          >
            {state.featureResult.summary}
          </div>
          {state.featureResult.metadata &&
            Object.keys(state.featureResult.metadata).length > 0 && (
              <div
                className='rounded-lg p-2 space-y-1'
                style={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border)",
                }}
              >
                {Object.entries(state.featureResult.metadata).map(([k, v]) => (
                  <div
                    key={k}
                    className='flex justify-between items-start gap-2 text-xs'
                    style={{ fontSize: 10 }}
                  >
                    <span style={{ color: "var(--fg-muted)" }}>{k}</span>
                    <span
                      className='text-right max-w-[70%] break-all whitespace-normal'
                      style={{ color: "var(--fg-secondary)" }}
                    >
                      {formatMetadataValue(v)}
                    </span>
                  </div>
                ))}
              </div>
            )}
        </>
      )}
    </div>
  );
}
