import { useState } from "react";
import {
  FileText,
  AudioLines,
  Search,
  BrainCircuit,
  Loader2,
  Mic,
} from "lucide-react";
import { useApp } from "../store/AppContext";
import SummarizeFeature from "./features/SummarizeFeature";
import AudioFeature from "./features/AudioFeature";
import SearchFeature from "./features/SearchFeature";
import XaiFeature from "./features/XaiFeature";
import TranscriptionFeature from "./features/TranscriptionFeature";

const TABS = [
  { id: "summarize", label: "Summarize", icon: FileText },
  { id: "transcribe", label: "Transcribe", icon: Mic },
  { id: "audio", label: "Audio", icon: AudioLines },
  { id: "search", label: "Search", icon: Search },
  { id: "xai", label: "Explain", icon: BrainCircuit },
];

const FEATURE_COMPONENTS = {
  summarize: SummarizeFeature,
  transcribe: TranscriptionFeature,
  audio: AudioFeature,
  search: SearchFeature,
  xai: XaiFeature,
};

export default function FeaturesPanel({ width = 320, onResizeStart }) {
  const { state, dispatch } = useApp();
  const [tab, setTab] = useState("transcribe");

  const selectTab = (id) => {
    // Don't allow selecting document-dependent features if no document is active
    if (!state.activeDocumentId && id !== "transcribe") {
      return;
    }
    setTab(id);
    dispatch({ type: "SET_ACTIVE_FEATURE", payload: id });
  };

  const FeatureComponent = FEATURE_COMPONENTS[tab] || TranscriptionFeature;
  const requiresDocument = tab !== "transcribe";

  return (
    <aside
      className='flex flex-col h-full border-l'
      style={{
        position: "relative",
        width,
        minWidth: width,
        maxWidth: width,
        flexShrink: 0,
        borderColor: "var(--border)",
        background: "var(--bg-surface)",
      }}
    >
      <div
        role='separator'
        aria-label='Resize features panel'
        aria-orientation='vertical'
        onMouseDown={(e) => {
          e.preventDefault();
          onResizeStart?.();
        }}
        className='absolute top-0 left-0 h-full w-1.5 cursor-col-resize'
        style={{ transform: "translateX(-50%)", zIndex: 20 }}
      />

      <div
        className='grid grid-cols-2 gap-1 p-2 border-b'
        style={{ borderColor: "var(--border)" }}
      >
        {TABS.map((tabItem) => {
          const TabIcon = tabItem.icon;
          const isDisabled =
            !state.activeDocumentId && tabItem.id !== "transcribe";

          return (
            <button
              key={tabItem.id}
              onClick={() => selectTab(tabItem.id)}
              disabled={isDisabled}
              className='flex items-center justify-center gap-2 py-2.5 rounded-lg text-xs transition-colors'
              style={{
                background:
                  tab === tabItem.id ? "var(--accent-muted)" : "transparent",
                color:
                  tab === tabItem.id ? "var(--accent)" : "var(--fg-tertiary)",
                opacity: isDisabled ? 0.4 : 1,
                cursor: isDisabled ? "not-allowed" : "pointer",
              }}
            >
              <TabIcon size={14} />
              {tabItem.label}
            </button>
          );
        })}
      </div>

      <div className='flex-1 overflow-y-auto p-4'>
        {state.featureError && (
          <p
            className='text-xs mb-3 px-2 py-1.5 rounded-md'
            style={{
              color: "var(--error)",
              background: "rgba(248,113,113,0.08)",
            }}
          >
            {state.featureError}
          </p>
        )}

        {state.isFeatureLoading && (
          <div className='flex items-center justify-center py-8'>
            <Loader2
              size={20}
              className='animate-spin'
              style={{ color: "var(--accent)" }}
            />
          </div>
        )}

        <div style={{ display: state.isFeatureLoading ? "none" : "block" }}>
          {!state.activeDocumentId && requiresDocument ? (
            <p
              className='text-xs text-center py-8'
              style={{ color: "var(--fg-muted)" }}
            >
              Select a document to use this feature
            </p>
          ) : (
            <FeatureComponent />
          )}
        </div>
      </div>
    </aside>
  );
}
