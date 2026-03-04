import { useEffect, useState } from "react";
import { Sun, Moon, User } from "lucide-react";
import SourcesPanel from "./components/SourcesPanel";
import ChatPanel from "./components/ChatPanel";
import FeaturesPanel from "./components/FeaturesPanel";
import { useApp } from "./store/AppContext";
import * as api from "./api/client";

export default function App() {
  const { dispatch } = useApp();
  const [theme, setTheme] = useState(() => localStorage.getItem("theme") || "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === "dark" ? "light" : "dark");
  };

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
            <img 
              src={theme === "light" ? "/logos/AuraLearn-Light.svg" : "/logos/AuraLearn-Dark.svg"} 
              alt="AuraLearn Logo" 
              className="w-5 h-5 object-contain"
            />
            <span
              className="text-base font-bold tracking-tight"
              style={{ color: "var(--fg-primary)" }}
            >
              AuraLearn
            </span>
          </div>
          <div className="flex items-center gap-2.5">
            <button
              onClick={toggleTheme}
              className="w-8 h-8 rounded-lg flex items-center justify-center"
              style={{
                background: "var(--bg-elevated)",
                color: "var(--fg-secondary)",
                border: "1px solid var(--border)",
              }}
              title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            >
              {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
            </button>
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
          </div>
        </nav>

        <div className="flex flex-1 min-h-0">
          <SourcesPanel />
          <ChatPanel />
          <FeaturesPanel />
        </div>
    </div>
  );
}
