import { useCallback, useState } from "react";
import { ChevronDown } from "lucide-react";
import { useApp } from "../../store/AppContext";
import * as api from "../../api/client";

export default function XaiFeature() {
  const { state, dispatch } = useApp();
  const [mode, setMode] = useState("extractive");
  const [numSentences, setNumSentences] = useState(3);
  const [maxLength, setMaxLength] = useState(150);
  const [minLength, setMinLength] = useState(50);
  const [generateLrp, setGenerateLrp] = useState(false);
  const [generateShap, setGenerateShap] = useState(false);

  const run = useCallback(async () => {
    dispatch({ type: "SET_FEATURE_LOADING", payload: true });
    try {
      let res;
      if (mode === "extractive") {
        res = await api.explainExtractive(state.activeDocumentId, {
          numSentences,
          generateLrp,
        });
      } else {
        res = await api.explainAbstractive(state.activeDocumentId, {
          maxLength,
          minLength,
          generateShap,
        });
      }
      dispatch({ type: "SET_FEATURE_RESULT", payload: res });
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    }
  }, [
    state.activeDocumentId,
    mode,
    numSentences,
    maxLength,
    minLength,
    generateLrp,
    generateShap,
    dispatch,
  ]);

  const r = state.featureResult;

  return (
    <div className='space-y-4'>
      <div
        className='flex gap-1 p-1 rounded-lg'
        style={{ background: "var(--bg-elevated)" }}
      >
        {["extractive", "abstractive"].map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className='flex-1 text-xs py-1.5 rounded-md transition-colors capitalize'
            style={{
              background: mode === m ? "var(--accent-muted)" : "transparent",
              color: mode === m ? "var(--accent)" : "var(--fg-tertiary)",
            }}
          >
            {m}
          </button>
        ))}
      </div>

      {mode === "extractive" && (
        <div className='space-y-3'>
          <div>
            <label
              className='text-xs mb-1 block'
              style={{ color: "var(--fg-secondary)" }}
            >
              Sentences: {numSentences}
            </label>
            <input
              type='range'
              min={1}
              max={10}
              value={numSentences}
              onChange={(e) => setNumSentences(Number(e.target.value))}
              className='w-full slider-styled'
            />
          </div>
          <label
            className='flex items-center gap-2 text-xs cursor-pointer'
            style={{ color: "var(--fg-secondary)" }}
          >
            <input
              type='checkbox'
              checked={generateLrp}
              onChange={(e) => setGenerateLrp(e.target.checked)}
              className="appearance-none w-4 h-4 m-0 rounded-full border border-(--accent) checked:bg-(--accent) checked:border-(--accent) cursor-pointer transition-colors bg-center bg-no-repeat checked:bg-[url('data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%223%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Cpolyline%20points=%2220%206%209%2017%204%2012%22/%3E%3C/svg%3E')] checked:bg-size-[65%_65%]"
            />
            Enable LRP Explanation (Advanced)
          </label>
        </div>
      )}

      {mode === "abstractive" && (
        <div className='space-y-3'>
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
          <label
            className='flex flex-col gap-1 text-xs cursor-pointer'
            style={{ color: "var(--fg-secondary)" }}
          >
            <div className='flex items-center gap-2'>
              <input
                type='checkbox'
                checked={generateShap}
                onChange={(e) => setGenerateShap(e.target.checked)}
                className="appearance-none w-4 h-4 m-0 rounded-full border border-(--accent) checked:bg-(--accent) checked:border-(--accent) cursor-pointer transition-colors bg-center bg-no-repeat checked:bg-[url('data:image/svg+xml,%3Csvg%20xmlns=%22http://www.w3.org/2000/svg%22%20viewBox=%220%200%2024%2024%22%20fill=%22none%22%20stroke=%22white%22%20stroke-width=%223%22%20stroke-linecap=%22round%22%20stroke-linejoin=%22round%22%3E%3Cpolyline%20points=%2220%206%209%2017%204%2012%22/%3E%3C/svg%3E')] checked:bg-size-[65%_65%]"
              />
              Enable SHAP Explanation (Advanced)
            </div>
            <span
              className='text-[10px] ml-6'
              style={{ color: "var(--fg-muted)" }}
            >
              Takes 30-60 seconds to compute
            </span>
          </label>
        </div>
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
        {state.isFeatureLoading ? "Analyzing…" : "Explain"}
      </button>

      {r?.xai_type?.includes("post-hoc + deep_explanation") && (
        <>
          <div className='grid grid-cols-2 gap-2'>
            {[
              ["Input sentences", r.num_sentences_input],
              ["Selected", r.num_sentences_selected],
              [
                "Avg selected",
                `${((r.average_score_selected || 0) * 100).toFixed(1)}%`,
              ],
              ["Avg all", `${((r.average_score_all || 0) * 100).toFixed(1)}%`],
            ].map(([label, val]) => (
              <div
                key={label}
                className='rounded-md p-2'
                style={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border)",
                }}
              >
                <p
                  className='text-xs'
                  style={{ color: "var(--fg-muted)", fontSize: 9 }}
                >
                  {label}
                </p>
                <p
                  className='text-sm font-medium'
                  style={{ color: "var(--fg-primary)" }}
                >
                  {val}
                </p>
              </div>
            ))}
          </div>

          {r.score_distribution &&
            Object.keys(r.score_distribution).length > 0 && (
              <div>
                <p
                  className='text-xs mb-2'
                  style={{ color: "var(--fg-muted)", fontSize: 10 }}
                >
                  Score Distribution
                </p>
                <div className='space-y-1'>
                  {Object.entries(r.score_distribution).map(
                    ([range, count]) => (
                      <div key={range} className='flex items-center gap-2'>
                        <span
                          className='text-xs w-16 shrink-0'
                          style={{
                            color: "var(--fg-tertiary)",
                            fontSize: 9,
                            fontFamily: "var(--font-mono)",
                          }}
                        >
                          {range}
                        </span>
                        <div
                          className='flex-1 h-2 rounded-full'
                          style={{ background: "var(--bg-elevated)" }}
                        >
                          <div
                            className='h-full rounded-full'
                            style={{
                              width: `${Math.min((count / (r.num_sentences_input || 1)) * 100, 100)}%`,
                              background: "var(--accent)",
                              opacity: 0.7,
                            }}
                          />
                        </div>
                        <span
                          className='text-xs w-4 text-right'
                          style={{ color: "var(--fg-muted)", fontSize: 9 }}
                        >
                          {count}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </div>
            )}

          {r.summary && (
            <div
              className='rounded-lg p-3 text-sm leading-relaxed'
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                color: "var(--fg-primary)",
              }}
            >
              {r.summary}
            </div>
          )}

          {r.sentences && (
            <div className='space-y-2'>
              <p
                className='text-xs'
                style={{ color: "var(--fg-muted)", fontSize: 10 }}
              >
                Sentence Analysis (Top 50 shown)
              </p>
              {r.sentences.slice(0, 50).map((s, i) => (
                <div
                  key={i}
                  className='rounded-lg p-2.5 space-y-1.5'
                  style={{
                    background: "var(--bg-elevated)",
                    border: `1px solid ${s.is_selected ? "var(--accent)" : "var(--border)"}`,
                    borderLeftWidth: 3,
                    borderLeftColor: s.is_selected
                      ? "var(--accent)"
                      : "var(--border)",
                  }}
                >
                  <div className='flex items-center gap-2 flex-wrap'>
                    <span
                      className='text-xs font-medium'
                      style={{
                        color: s.is_selected
                          ? "var(--accent)"
                          : "var(--fg-secondary)",
                        fontSize: 10,
                      }}
                    >
                      S{s.index + 1}
                    </span>
                    {s.is_selected && (
                      <span
                        className='text-xs px-1 py-0.5 rounded'
                        style={{
                          background: "var(--accent-muted)",
                          color: "var(--accent)",
                          fontSize: 8,
                        }}
                      >
                        SELECTED
                      </span>
                    )}
                    <span
                      className='text-xs'
                      style={{ color: "var(--fg-muted)", fontSize: 9 }}
                    >
                      {s.word_count} words
                    </span>
                  </div>
                  <p
                    className='text-xs leading-relaxed'
                    style={{ color: "var(--fg-secondary)" }}
                  >
                    {s.text}
                  </p>
                  <div className='flex items-center gap-2'>
                    <div
                      className='flex-1 h-1.5 rounded-full'
                      style={{ background: "var(--bg-overlay)" }}
                    >
                      <div
                        className='h-full rounded-full'
                        style={{
                          width: `${Math.max((s.importance_score || 0) * 100, 2)}%`,
                          background: s.is_selected
                            ? "var(--accent)"
                            : "var(--fg-muted)",
                        }}
                      />
                    </div>
                    <span
                      className='text-xs shrink-0'
                      style={{
                        color: s.is_selected
                          ? "var(--accent)"
                          : "var(--fg-tertiary)",
                        fontSize: 9,
                      }}
                    >
                      {((s.importance_score || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div
                    className='flex gap-3 flex-wrap'
                    style={{ fontSize: 9, color: "var(--fg-muted)" }}
                  >
                    <span>Sensitivity: {s.sensitivity?.toFixed(3) ?? "—"}</span>
                    {s.most_attended_sentence != null && (
                      <span>Attends → S{s.most_attended_sentence + 1}</span>
                    )}
                  </div>
                  {s.attention_to_others &&
                    s.attention_to_others.length > 0 && (
                      <div className='flex gap-0.5 flex-wrap'>
                        {s.attention_to_others
                          .map((a, j) => ({ a, j }))
                          .filter(({ a }) => a > 0.05)
                          .slice(0, 10)
                          .map(({ a, j }) => (
                            <span
                              key={j}
                              className='px-1 py-0.5 rounded text-xs'
                              style={{
                                fontSize: 8,
                                background: `rgba(78, 197, 137, ${Math.min(a, 1) * 0.4})`,
                                color:
                                  a > 0.3 ? "var(--accent)" : "var(--fg-muted)",
                              }}
                            >
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
            <div className='flex gap-1 flex-wrap'>
              {r.explanation_methods.map((m) => (
                <span
                  key={m}
                  className='text-xs px-1.5 py-0.5 rounded'
                  style={{
                    background: "var(--bg-elevated)",
                    color: "var(--fg-muted)",
                    fontSize: 9,
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {m}
                </span>
              ))}
            </div>
          )}
        </>
      )}

      {r?.xai_type?.includes("post-hoc_sensitivity") && (
        <>
          {r.summary && (
            <div
              className='rounded-lg p-3 text-sm leading-relaxed'
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                color: "var(--fg-primary)",
              }}
            >
              {r.summary}
            </div>
          )}

          <div
            className='flex gap-2 flex-wrap'
            style={{ fontSize: 10, color: "var(--fg-muted)" }}
          >
            <span
              className='px-1.5 py-0.5 rounded'
              style={{ background: "var(--bg-elevated)" }}
            >
              {r.summary_word_count} tokens
            </span>
            <span
              className='px-1.5 py-0.5 rounded'
              style={{ background: "var(--bg-elevated)" }}
            >
              {((r.compression_ratio || 0) * 100).toFixed(1)}% compression
            </span>
          </div>

          {r.sentence_contributions && r.sentence_contributions.length > 0 && (
            <div>
              <p
                className='text-xs mb-2'
                style={{ color: "var(--fg-muted)", fontSize: 10 }}
              >
                Source Sentence Contributions (Top 50 shown)
              </p>
              <div className='space-y-2'>
                {r.sentence_contributions.slice(0, 50).map((sc, i) => {
                  const isMost =
                    r.most_influential_sentence?.index === sc.index;
                  return (
                    <div
                      key={i}
                      className='rounded-lg p-2.5 space-y-1'
                      style={{
                        background: "var(--bg-elevated)",
                        border: `1px solid ${isMost ? "var(--accent)" : "var(--border)"}`,
                        borderLeftWidth: 3,
                        borderLeftColor: isMost
                          ? "var(--accent)"
                          : "var(--border)",
                      }}
                    >
                      <div className='flex items-center gap-2'>
                        <span
                          className='text-xs font-medium'
                          style={{
                            color: isMost
                              ? "var(--accent)"
                              : "var(--fg-secondary)",
                            fontSize: 10,
                          }}
                        >
                          S{sc.index + 1}
                        </span>
                        {isMost && (
                          <span
                            className='text-xs px-1 py-0.5 rounded'
                            style={{
                              background: "var(--accent-muted)",
                              color: "var(--accent)",
                              fontSize: 8,
                            }}
                          >
                            MOST INFLUENTIAL
                          </span>
                        )}
                      </div>
                      <p
                        className='text-xs leading-relaxed'
                        style={{ color: "var(--fg-secondary)" }}
                      >
                        {sc.text}
                      </p>
                      <div className='flex items-center gap-2'>
                        <div
                          className='flex-1 h-1.5 rounded-full'
                          style={{ background: "var(--bg-overlay)" }}
                        >
                          <div
                            className='h-full rounded-full'
                            style={{
                              width: `${Math.max((sc.normalized_contribution || sc.contribution_score || 0) * 100, 2)}%`,
                              background: isMost
                                ? "var(--accent)"
                                : "var(--fg-muted)",
                            }}
                          />
                        </div>
                        <span
                          className='text-xs shrink-0'
                          style={{
                            color: isMost
                              ? "var(--accent)"
                              : "var(--fg-tertiary)",
                            fontSize: 9,
                          }}
                        >
                          {(
                            (sc.normalized_contribution ||
                              sc.contribution_score ||
                              0) * 100
                          ).toFixed(1)}
                          %
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
              <p
                className='text-xs mb-2'
                style={{ color: "var(--fg-muted)", fontSize: 10 }}
              >
                Token Confidence (First 300 tokens)
              </p>
              <div className='flex flex-wrap gap-0.5'>
                {r.token_confidence.slice(0, 300).map((t, i) => (
                  <span
                    key={i}
                    className='text-xs px-1 py-0.5 rounded cursor-default'
                    title={`${((t.confidence || 0) * 100).toFixed(1)}% confidence`}
                    style={{
                      fontSize: 10,
                      background: `rgba(78, 197, 137, ${Math.min(t.confidence, 1) * 0.3})`,
                      color: t.is_high_confidence
                        ? "var(--accent)"
                        : "var(--fg-tertiary)",
                      borderBottom: t.is_high_confidence
                        ? "1px solid var(--accent)"
                        : "1px solid transparent",
                    }}
                  >
                    {t.token}
                  </span>
                ))}
              </div>
            </div>
          )}

          {r.original_text_sentences &&
            r.original_text_sentences.length > 0 && (
              <details className='group'>
                <summary
                  className='text-xs cursor-pointer list-none flex items-center gap-1'
                  style={{ color: "var(--fg-muted)", fontSize: 10 }}
                >
                  <ChevronDown
                    size={12}
                    className='group-open:rotate-180 transition-transform'
                  />
                  Source text ({r.original_text_sentences.length} sentences)
                </summary>
                <div
                  className='mt-2 rounded-lg p-2 space-y-1'
                  style={{
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                  }}
                >
                  {r.original_text_sentences.map((s, i) => (
                    <p
                      key={i}
                      className='text-xs leading-relaxed'
                      style={{ color: "var(--fg-secondary)", fontSize: 11 }}
                    >
                      <span
                        style={{
                          color: "var(--fg-muted)",
                          fontFamily: "var(--font-mono)",
                          fontSize: 9,
                        }}
                      >
                        {i + 1}.{" "}
                      </span>
                      {s}
                    </p>
                  ))}
                </div>
              </details>
            )}

          {r.explanation_methods && (
            <div className='flex gap-1 flex-wrap'>
              {r.explanation_methods.map((m) => (
                <span
                  key={m}
                  className='text-xs px-1.5 py-0.5 rounded'
                  style={{
                    background: "var(--bg-elevated)",
                    color: "var(--fg-muted)",
                    fontSize: 9,
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {m}
                </span>
              ))}
            </div>
          )}
        </>
      )}

      {r && !r.xai_type && r.summary && (
        <div
          className='rounded-lg p-3 text-sm leading-relaxed'
          style={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            color: "var(--fg-primary)",
          }}
        >
          {r.summary}
        </div>
      )}
    </div>
  );
}
