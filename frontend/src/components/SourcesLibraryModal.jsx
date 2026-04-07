import { useMemo, useState } from "react";
import {
  AudioLines,
  FileText,
  Loader2,
  Video,
  X,
} from "lucide-react";
import * as api from "../api/client";

const MEDIA_AUDIO = new Set(["mp3", "wav", "m4a", "mpga"]);
const MEDIA_VIDEO = new Set(["mp4", "webm", "mpeg"]);

const formatBytes = (bytes = 0) => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
};

const sourceTypeIcon = (source) => {
  const type = (source.file_type || "").toLowerCase();
  if (MEDIA_VIDEO.has(type)) return Video;
  if (MEDIA_AUDIO.has(type)) return AudioLines;
  return FileText;
};

export default function SourcesLibraryModal({ isOpen, onClose, sessionId, sources = [] }) {
  const [openingId, setOpeningId] = useState(null);
  const [error, setError] = useState("");

  const sortedSources = useMemo(() => {
    return [...sources].sort((a, b) => {
      const aTime = new Date(a.added_at || 0).getTime();
      const bTime = new Date(b.added_at || 0).getTime();
      return bTime - aTime;
    });
  }, [sources]);

  if (!isOpen) return null;

  const openSource = async (source) => {
    if (!sessionId) return;

    setOpeningId(source.source_id);
    setError("");

    try {
      const file = await api.getSourceFileBlob(sessionId, source.source_id);
      const blob = new Blob([file.blob], { type: file.contentType });
      const url = URL.createObjectURL(blob);

      const popup = window.open(url, "_blank", "noopener,noreferrer");
      if (!popup) {
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
        anchor.download = file.filename;
        anchor.click();
      }

      window.setTimeout(() => URL.revokeObjectURL(url), 120000);
    } catch (err) {
      setError(err.message);
    } finally {
      setOpeningId(null);
    }
  };

  return (
    <div
      className='fixed inset-0 z-50 flex items-center justify-center px-4'
      style={{ background: "rgba(10,10,11,0.64)" }}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose?.();
      }}
    >
      <div
        className='w-full max-w-xl rounded-2xl border shadow-2xl overflow-hidden'
        style={{
          background: "var(--bg-surface)",
          borderColor: "var(--border)",
        }}
      >
        <div
          className='flex items-center justify-between px-5 py-4 border-b'
          style={{ borderColor: "var(--border)" }}
        >
          <div>
            <h3 className='text-sm font-semibold' style={{ color: "var(--fg-primary)" }}>
              Uploaded Sources
            </h3>
            <p className='text-xs mt-0.5' style={{ color: "var(--fg-muted)" }}>
              {sortedSources.length} file{sortedSources.length === 1 ? "" : "s"} in this chat
            </p>
          </div>

          <button
            onClick={onClose}
            className='w-8 h-8 rounded-lg flex items-center justify-center'
            style={{
              background: "var(--bg-elevated)",
              color: "var(--fg-secondary)",
              border: "1px solid var(--border)",
            }}
          >
            <X size={14} />
          </button>
        </div>

        <div className='px-5 py-4'>
          {sortedSources.length === 0 ? (
            <div
              className='rounded-lg px-4 py-6 text-center'
              style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}
            >
              <p className='text-xs' style={{ color: "var(--fg-secondary)" }}>
                No sources uploaded for this chat yet.
              </p>
            </div>
          ) : (
            <div className='space-y-2 max-h-96 overflow-y-auto pr-1'>
              {sortedSources.map((source) => {
                const Icon = sourceTypeIcon(source);
                const isOpening = openingId === source.source_id;
                return (
                  <button
                    key={source.source_id}
                    onClick={() => openSource(source)}
                    disabled={isOpening}
                    className='w-full flex items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-colors'
                    style={{
                      background: "var(--bg-elevated)",
                      border: "1px solid var(--border)",
                      opacity: isOpening ? 0.8 : 1,
                    }}
                  >
                    <div
                      className='w-8 h-8 rounded-lg flex items-center justify-center shrink-0'
                      style={{ background: "var(--accent-muted)" }}
                    >
                      <Icon size={14} style={{ color: "var(--accent)" }} />
                    </div>

                    <div className='min-w-0 flex-1'>
                      <p className='text-xs truncate' style={{ color: "var(--fg-primary)" }}>
                        {source.filename}
                      </p>
                      <p className='text-[10px] mt-0.5' style={{ color: "var(--fg-muted)" }}>
                        {String(source.file_type || "file").toUpperCase()} • {formatBytes(source.size_bytes || 0)}
                      </p>
                    </div>

                    {isOpening ? (
                      <Loader2 size={14} className='animate-spin shrink-0' style={{ color: "var(--accent)" }} />
                    ) : (
                      <span className='text-[10px] shrink-0' style={{ color: "var(--fg-tertiary)" }}>
                        Preview
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {error && (
            <p className='text-xs mt-3' style={{ color: "var(--error)" }}>
              {error}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
