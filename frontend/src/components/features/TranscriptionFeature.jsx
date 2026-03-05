import { useCallback, useState, useRef } from "react";
import { Upload, FileAudio, FileVideo, X } from "lucide-react";
import { useApp } from "../../store/AppContext";
import * as api from "../../api/client";

export default function TranscriptionFeature() {
  const { dispatch } = useApp();
  const [selectedFile, setSelectedFile] = useState(null);
  const [type, setType] = useState("extractive");
  const [sentences, setSentences] = useState(3);
  const [maxLength, setMaxLength] = useState(150);
  const [minLength, setMinLength] = useState(40);
  const [isProcessing, setIsProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      setResult(null);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setResult(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const run = useCallback(async () => {
    if (!selectedFile) return;

    setIsProcessing(true);
    dispatch({ type: "SET_FEATURE_ERROR", payload: null });

    try {
      const res = await api.transcribeMedia(selectedFile, {
        type,
        numSentences: type === "extractive" ? sentences : undefined,
        maxLength: type === "abstractive" ? maxLength : undefined,
        minLength: type === "abstractive" ? minLength : undefined,
      });
      setResult(res);
    } catch (err) {
      dispatch({ type: "SET_FEATURE_ERROR", payload: err.message });
    } finally {
      setIsProcessing(false);
    }
  }, [selectedFile, type, sentences, maxLength, minLength, dispatch]);

  const getFileIcon = () => {
    if (!selectedFile) return null;
    const isVideo = selectedFile.type.startsWith("video/");
    return isVideo ? <FileVideo size={16} /> : <FileAudio size={16} />;
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  return (
    <div className='space-y-4'>
      <div>
        <label
          className='text-xs mb-2 block'
          style={{ color: "var(--fg-secondary)" }}
        >
          Upload Audio/Video
        </label>
        <input
          ref={fileInputRef}
          type='file'
          accept='audio/*,video/*,.mp3,.wav,.mp4,.m4a,.mpeg,.mpga,.webm'
          onChange={handleFileSelect}
          className='hidden'
        />
        {!selectedFile ? (
          <button
            onClick={() => fileInputRef.current?.click()}
            className='w-full py-8 rounded-lg border-2 border-dashed transition-colors flex flex-col items-center gap-2'
            style={{
              borderColor: "var(--border)",
              background: "var(--bg-elevated)",
              color: "var(--fg-secondary)",
            }}
          >
            <Upload size={24} style={{ color: "var(--accent)" }} />
            <span className='text-xs'>Click to select file</span>
            <span className='text-[10px]' style={{ color: "var(--fg-muted)" }}>
              MP3, WAV, MP4, M4A, WebM
            </span>
          </button>
        ) : (
          <div
            className='flex items-center gap-2 p-3 rounded-lg'
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
            }}
          >
            <div
              className='flex items-center justify-center w-8 h-8 rounded shrink-0'
              style={{
                background: "var(--accent-muted)",
                color: "var(--accent)",
              }}
            >
              {getFileIcon()}
            </div>
            <div className='flex-1 min-w-0'>
              <div
                className='text-xs truncate'
                style={{ color: "var(--fg-primary)" }}
              >
                {selectedFile.name}
              </div>
              <div className='text-[10px]' style={{ color: "var(--fg-muted)" }}>
                {formatFileSize(selectedFile.size)}
              </div>
            </div>
            <button
              onClick={clearFile}
              className='w-6 h-6 rounded flex items-center justify-center shrink-0 transition-colors'
              style={{
                background: "var(--bg-surface)",
                color: "var(--fg-muted)",
              }}
            >
              <X size={14} />
            </button>
          </div>
        )}
      </div>

      {selectedFile && (
        <>
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
                  background:
                    type === t ? "var(--accent-muted)" : "transparent",
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
                Summary sentences: {sentences}
              </label>
              <input
                type='range'
                min={1}
                max={10}
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
            disabled={isProcessing}
            className='w-full py-2 rounded-lg text-xs font-medium transition-colors'
            style={{
              background: "var(--accent)",
              color: "#fff",
              opacity: isProcessing ? 0.5 : 1,
            }}
          >
            {isProcessing ? "Transcribing…" : "Transcribe & Summarize"}
          </button>
        </>
      )}

      {result && (
        <div className='space-y-3 pt-2'>
          <div
            className='flex items-center gap-2 flex-wrap text-[10px]'
            style={{ color: "var(--fg-muted)" }}
          >
            <span
              className='px-1.5 py-0.5 rounded'
              style={{ background: "var(--bg-elevated)" }}
            >
              {result.summarization_type}
            </span>
            <span
              className='px-1.5 py-0.5 rounded'
              style={{ background: "var(--bg-elevated)" }}
            >
              {result.language}
            </span>
            {result.metadata?.segments_count && (
              <span
                className='px-1.5 py-0.5 rounded'
                style={{ background: "var(--bg-elevated)" }}
              >
                {result.metadata.segments_count} segments
              </span>
            )}
          </div>

          <div>
            <label
              className='text-xs mb-1.5 block font-medium'
              style={{ color: "var(--fg-secondary)" }}
            >
              Summary
            </label>
            <div
              className='rounded-lg p-3 text-sm leading-relaxed'
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                color: "var(--fg-primary)",
              }}
            >
              {result.summary}
            </div>
          </div>

          <div>
            <label
              className='text-xs mb-1.5 block font-medium'
              style={{ color: "var(--fg-secondary)" }}
            >
              Full Transcription
            </label>
            <div
              className='rounded-lg p-3 text-sm leading-relaxed max-h-64 overflow-y-auto'
              style={{
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                color: "var(--fg-primary)",
              }}
            >
              {result.text}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
