import { useEffect, useRef, useState } from "react";
import { Sun, Moon, User } from "lucide-react";
import SourcesPanel from "./components/SourcesPanel";
import ChatPanel from "./components/ChatPanel";
import FeaturesPanel from "./components/FeaturesPanel";
import { useApp } from "./store/AppContext";
import * as api from "./api/client";

const SOURCE_WIDTH_KEY = "sources-panel-width";
const FEATURE_WIDTH_KEY = "features-panel-width";
const DEFAULT_SOURCE_WIDTH = 280;
const DEFAULT_FEATURE_WIDTH = 320;
const MIN_SOURCE_WIDTH = 220;
const MAX_SOURCE_WIDTH = 460;
const MIN_FEATURE_WIDTH = 240;
const MAX_FEATURE_WIDTH = 520;
const MIN_CENTER_WIDTH = 420;

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const getStoredWidth = (key, fallback) => {
  const raw = localStorage.getItem(key);
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
};

export default function App() {
  const { dispatch } = useApp();
  const [theme, setTheme] = useState(
    () => localStorage.getItem("theme") || "dark",
  );
  const [sourcesWidth, setSourcesWidth] = useState(() =>
    getStoredWidth(SOURCE_WIDTH_KEY, DEFAULT_SOURCE_WIDTH),
  );
  const [featuresWidth, setFeaturesWidth] = useState(() =>
    getStoredWidth(FEATURE_WIDTH_KEY, DEFAULT_FEATURE_WIDTH),
  );
  const [resizingPanel, setResizingPanel] = useState(null);
  const sourcesWidthRef = useRef(sourcesWidth);
  const featuresWidthRef = useRef(featuresWidth);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem(SOURCE_WIDTH_KEY, String(sourcesWidth));
    sourcesWidthRef.current = sourcesWidth;
  }, [sourcesWidth]);

  useEffect(() => {
    localStorage.setItem(FEATURE_WIDTH_KEY, String(featuresWidth));
    featuresWidthRef.current = featuresWidth;
  }, [featuresWidth]);

  useEffect(() => {
    if (!resizingPanel) return;

    const handleMouseMove = (e) => {
      const viewportWidth = window.innerWidth;

      if (resizingPanel === "sources") {
        const dynamicMax = Math.min(
          MAX_SOURCE_WIDTH,
          viewportWidth - featuresWidthRef.current - MIN_CENTER_WIDTH,
        );
        if (dynamicMax < MIN_SOURCE_WIDTH) return;
        const nextWidth = clamp(e.clientX, MIN_SOURCE_WIDTH, dynamicMax);
        setSourcesWidth(nextWidth);
        return;
      }

      const rawFeatureWidth = viewportWidth - e.clientX;
      const dynamicMax = Math.min(
        MAX_FEATURE_WIDTH,
        viewportWidth - sourcesWidthRef.current - MIN_CENTER_WIDTH,
      );
      if (dynamicMax < MIN_FEATURE_WIDTH) return;
      const nextWidth = clamp(rawFeatureWidth, MIN_FEATURE_WIDTH, dynamicMax);
      setFeaturesWidth(nextWidth);
    };

    const handleMouseUp = () => {
      setResizingPanel(null);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };

    document.body.style.userSelect = "none";
    document.body.style.cursor = "col-resize";
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [resizingPanel]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === "dark" ? "light" : "dark"));
  };

  useEffect(() => {
    api
      .listDocuments()
      .then((res) =>
        dispatch({ type: "SET_DOCUMENTS", payload: res.documents || [] }),
      )
      .catch(() => {});
  }, [dispatch]);

  return (
    <div
      className='flex flex-col h-full'
      style={{ background: "var(--bg-base)" }}
    >
      <nav
        className='flex items-center justify-between px-5 py-2.5 border-b shrink-0'
        style={{
          borderColor: "var(--border)",
          background: "var(--bg-surface)",
        }}
      >
        <div className='flex items-center gap-2.5'>
          <img
            src={
              theme === "light"
                ? "/logos/AuraLearn-Light.svg"
                : "/logos/AuraLearn-Dark.svg"
            }
            alt='AuraLearn Logo'
            className='w-5 h-5 object-contain'
          />
          <span
            className='text-base font-bold tracking-tight'
            style={{ color: "var(--fg-primary)" }}
          >
            AuraLearn
          </span>
        </div>
        <div className='flex items-center gap-2.5'>
          <button
            onClick={toggleTheme}
            className='w-8 h-8 rounded-lg flex items-center justify-center'
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
            className='flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs transition-colors'
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

      <div className='flex flex-1 min-h-0'>
        <SourcesPanel
          width={sourcesWidth}
          onResizeStart={() => setResizingPanel("sources")}
        />
        <ChatPanel />
        <FeaturesPanel
          width={featuresWidth}
          onResizeStart={() => setResizingPanel("features")}
        />
      </div>
    </div>
  );
}
