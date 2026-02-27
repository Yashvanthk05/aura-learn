import { useCallback, useEffect, useRef, useState } from "react";
import { Send, Loader2, Bot, User, Sparkles } from "lucide-react";
import { useApp } from "../store/AppContext";
import * as api from "../api/client";

export default function ChatPanel() {
  const { state, dispatch } = useApp();
  const [input, setInput] = useState("");
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });

  useEffect(() => {
    scrollToBottom();
  }, [state.messages]);

  useEffect(() => {
    if (state.activeDocumentId && !state.sessionId) {
      api
        .createChatSession(state.activeDocumentId)
        .then((res) => dispatch({ type: "SET_SESSION", payload: res.session_id }))
        .catch(() => {});
    }
  }, [state.activeDocumentId, state.sessionId, dispatch]);

  const handleSend = useCallback(async () => {
    const q = input.trim();
    if (!q || !state.sessionId || state.isChatLoading) return;

    setInput("");
    dispatch({
      type: "ADD_MESSAGE",
      payload: { role: "user", content: q, timestamp: new Date().toISOString() },
    });
    dispatch({ type: "SET_CHAT_LOADING", payload: true });

    try {
      const res = await api.chatQuery(state.sessionId, q);
      dispatch({
        type: "ADD_MESSAGE",
        payload: {
          role: "assistant",
          content: res.response,
          citations: res.citations,
          timestamp: res.timestamp,
        },
      });
      dispatch({ type: "SET_CHAT_LOADING", payload: false });
    } catch (err) {
      dispatch({ type: "SET_CHAT_ERROR", payload: err.message });
    }
  }, [input, state.sessionId, state.isChatLoading, dispatch]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (!state.activeDocumentId) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-4 p-8">
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center"
          style={{ background: "var(--accent-muted)" }}
        >
          <Sparkles size={28} style={{ color: "var(--accent)" }} />
        </div>
        <div className="text-center max-w-sm">
          <h2
            className="text-lg font-medium mb-2"
            style={{ color: "var(--fg-primary)" }}
          >
            Welcome to AuraLearn
          </h2>
          <p
            className="text-sm leading-relaxed"
            style={{ color: "var(--fg-secondary)", fontSize: 12 }}
          >
            Upload a PDF source from the left panel to start chatting, summarizing, and generating audio.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col min-w-0">
      <div
        className="flex items-center gap-3 px-5 py-3 border-b shrink-0"
        style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
      >
        <Bot size={16} style={{ color: "var(--accent)" }} />
        <span
          className="text-sm font-medium"
          style={{ color: "var(--fg-primary)" }}
        >
          Chat
        </span>
        {state.activeDocument && (
          <span
            className="text-xs px-2 py-0.5 rounded-md"
            style={{
              color: "var(--fg-tertiary)",
              background: "var(--bg-elevated)",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
            }}
          >
            {state.activeDocument.filename}
          </span>
        )}
      </div>

      <div
        className="flex-1 overflow-y-auto px-5 py-4 space-y-4"
        style={{ background: "transparent" }}
      >
        {state.messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full gap-3">
            <p
              className="text-sm text-center"
              style={{ color: "var(--fg-muted)", fontSize: 12 }}
            >
              Ask anything about your document
            </p>
          </div>
        )}

        {state.messages.map((msg, i) => (
          <div
            key={i}
            className={`flex gap-3 ${msg.role === "user" ? "justify-end" : ""}`}
          >
            {msg.role === "assistant" && (
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-1"
                style={{ background: "var(--accent-muted)" }}
              >
                <Bot size={14} style={{ color: "var(--accent)" }} />
              </div>
            )}
            <div
              className="max-w-[75%] rounded-xl px-4 py-3"
              style={{
                background:
                  msg.role === "user"
                    ? "var(--accent-muted)"
                    : "var(--bg-elevated)",
                border: "1px solid",
                borderColor:
                  msg.role === "user"
                    ? "var(--accent-muted)"
                    : "var(--border)",
              }}
            >
              <p
                className="text-sm leading-relaxed whitespace-pre-wrap"
                style={{
                  color: "var(--fg-primary)",
                }}
              >
                {msg.content}
              </p>
              {msg.citations && msg.citations.length > 0 && (
                <div className="mt-2 pt-2 border-t" style={{ borderColor: "var(--border)" }}>
                  <p
                    className="text-xs mb-1"
                    style={{ color: "var(--fg-muted)", fontSize: 10 }}
                  >
                    Sources:
                  </p>
                  {msg.citations.map((c, j) => (
                    <div key={j} className="py-0.5" style={{ fontSize: 10 }}>
                      <div className="flex items-center gap-1.5 flex-wrap">
                        <span style={{ color: "var(--fg-secondary)" }}>[{j + 1}]</span>
                        {c.topic && <span className="font-medium" style={{ color: "var(--fg-secondary)" }}>{c.topic}</span>}
                        <span style={{ color: "var(--fg-tertiary)" }}>p.{c.page}</span>
                        {c.score != null && (
                          <span className="px-1 rounded" style={{ background: "var(--accent-muted)", color: "var(--accent)", fontSize: 8 }}>
                            {(c.score * 100).toFixed(0)}%
                          </span>
                        )}
                        {c.relevance && (
                          <span className="px-1 rounded" style={{ background: "var(--bg-overlay)", color: "var(--fg-muted)", fontSize: 8 }}>
                            {c.relevance}
                          </span>
                        )}
                      </div>
                      <p className="text-xs mt-0.5" style={{ color: "var(--fg-muted)" }}>
                        {(c.text_snippet || c.text || "").slice(0, 120)}…
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {msg.role === "user" && (
              <div
                className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-1"
                style={{ background: "var(--bg-overlay)" }}
              >
                <User size={14} style={{ color: "var(--fg-secondary)" }} />
              </div>
            )}
          </div>
        ))}

        {state.isChatLoading && (
          <div className="flex gap-3">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center shrink-0"
              style={{ background: "var(--accent-muted)" }}
            >
              <Loader2
                size={14}
                className="animate-spin"
                style={{ color: "var(--accent)" }}
              />
            </div>
            <div
              className="rounded-xl px-4 py-3"
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
              }}
            >
              <div className="flex gap-1">
                <span
                  className="w-1.5 h-1.5 rounded-full animate-bounce"
                  style={{ background: "var(--fg-muted)", animationDelay: "0ms" }}
                />
                <span
                  className="w-1.5 h-1.5 rounded-full animate-bounce"
                  style={{ background: "var(--fg-muted)", animationDelay: "150ms" }}
                />
                <span
                  className="w-1.5 h-1.5 rounded-full animate-bounce"
                  style={{ background: "var(--fg-muted)", animationDelay: "300ms" }}
                />
              </div>
            </div>
          </div>
        )}

        {state.chatError && (
          <p
            className="text-xs px-2"
            style={{ color: "var(--error)" }}
          >
            {state.chatError}
          </p>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div
        className="px-4 py-3 border-t shrink-0"
        style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
      >
        <div
          className="flex items-end gap-2 rounded-xl px-4 py-2"
          style={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
          }}
        >
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your document..."
            rows={1}
            className="flex-1 bg-transparent resize-none outline-none text-sm py-1"
            style={{
              color: "var(--fg-primary)",
              maxHeight: 120,
            }}
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || state.isChatLoading}
            className="p-2 rounded-lg transition-colors duration-150 shrink-0"
            style={{
              background:
                input.trim() && !state.isChatLoading
                  ? "var(--accent)"
                  : "var(--bg-overlay)",
              color:
                input.trim() && !state.isChatLoading
                  ? "#fff"
                  : "var(--fg-muted)",
              cursor:
                input.trim() && !state.isChatLoading
                  ? "pointer"
                  : "not-allowed",
            }}
          >
            <Send size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
