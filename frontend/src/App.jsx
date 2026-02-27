import { useEffect } from "react";
import { Sparkles, User } from "lucide-react";
import SourcesPanel from "./components/SourcesPanel";
import ChatPanel from "./components/ChatPanel";
import FeaturesPanel from "./components/FeaturesPanel";
import { useApp } from "./store/AppContext";
import * as api from "./api/client";

export default function App() {
  const { dispatch } = useApp();

  useEffect(() => {
    api
      .listDocuments()
      .then((res) =>
        dispatch({ type: "SET_DOCUMENTS", payload: res.documents || [] })
      )
      .catch(() => {});
  }, [dispatch]);

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: "var(--bg-base)" }}
    >
        <nav
          className="flex items-center justify-between px-5 py-2.5 border-b shrink-0"
          style={{ borderColor: "var(--border)", background: "var(--bg-surface)" }}
        >
          <div className="flex items-center gap-2.5">
            <div
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{ background: "var(--accent-muted)" }}
            >
              <Sparkles size={16} style={{ color: "var(--accent)" }} />
            </div>
            <span
              className="text-base font-bold tracking-tight"
              style={{ color: "var(--fg-primary)" }}
            >
              AuraLearn
            </span>
          </div>
          <button
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors"
            style={{
              background: "var(--bg-elevated)",
              color: "var(--fg-secondary)",
              border: "1px solid var(--border)",
            }}
          >
            <User size={14} />
            Log in
          </button>
        </nav>

        <div className="flex flex-1 min-h-0">
          <SourcesPanel />
          <ChatPanel />
          <FeaturesPanel />
        </div>
    </div>
  );
}
