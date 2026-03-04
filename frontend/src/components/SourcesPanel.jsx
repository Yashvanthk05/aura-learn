import { useCallback, useRef, useState } from "react";
import { Upload, FileText, X, Loader2, ChevronRight } from "lucide-react";
import { useApp } from "../store/AppContext";
import * as api from "../api/client";

export default function SourcesPanel() {
  const { state, dispatch } = useApp();
  const fileInputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleUpload = useCallback(
    async (files) => {
      const fileList = Array.from(files);
      const pdfs = fileList.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
      if (pdfs.length === 0) {
        dispatch({
          type: "SET_UPLOAD_ERROR",
          payload: "Only PDF files are supported.",
        });
        return;
      }

      dispatch({ type: "SET_UPLOADING", payload: true });
      let lastDoc = null;
      const errors = [];

      for (const file of pdfs) {
        try {
          const result = await api.uploadDocument(file);
          dispatch({ type: "ADD_DOCUMENT", payload: result });
          lastDoc = result;
        } catch (err) {
          errors.push(`${file.name}: ${err.message}`);
        }
      }

      dispatch({ type: "SET_UPLOADING", payload: false });

      if (errors.length > 0) {
        dispatch({ type: "SET_UPLOAD_ERROR", payload: errors.join("; ") });
      }

      // auto-select the last successfully uploaded doc
      if (lastDoc) {
        try {
          const docInfo = await api.getDocument(lastDoc.document_id);
          dispatch({
            type: "SET_ACTIVE_DOCUMENT",
            payload: { id: lastDoc.document_id, data: docInfo },
          });
        } catch {
          /* ignore */
        }
      }
    },
    [dispatch]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      if (e.dataTransfer.files.length > 0) {
        handleUpload(e.dataTransfer.files);
      }
    },
    [handleUpload]
  );

  const handleFileChange = useCallback(
    (e) => {
      if (e.target.files.length > 0) {
        handleUpload(e.target.files);
      }
      e.target.value = "";
    },
    [handleUpload]
  );

  const selectDocument = useCallback(
    async (docId) => {
      if (state.activeDocumentId === docId) return;
      try {
        const docInfo = await api.getDocument(docId);
        dispatch({
          type: "SET_ACTIVE_DOCUMENT",
          payload: { id: docId, data: docInfo },
        });
      } catch {
        /* ignore */
      }
    },
    [dispatch, state.activeDocumentId]
  );

  const removeDocument = useCallback(
    async (e, docId) => {
      e.stopPropagation();
      try {
        await api.deleteDocument(docId);
        dispatch({ type: "REMOVE_DOCUMENT", payload: docId });
      } catch {
        /* ignore */
      }
    },
    [dispatch]
  );

  return (
    <aside
      className="flex flex-col h-full border-r"
      style={{
        width: 280,
        minWidth: 280,
        borderColor: "var(--border)",
        background: "var(--bg-surface)",
      }}
    >
      <div
        className="flex items-center gap-2 px-4 py-3 border-b"
        style={{ borderColor: "var(--border)" }}
      >
        <div
          className="w-7 h-7 rounded-lg flex items-center justify-center"
          style={{ background: "var(--accent-muted)" }}
        >
          <FileText size={14} style={{ color: "var(--accent)" }} />
        </div>
        <span className="font-medium text-sm" style={{ color: "var(--fg-primary)" }}>
          Sources
        </span>
      </div>

      <div className="p-3">
        <div
          role="button"
          tabIndex={0}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
          className="flex flex-col items-center justify-center gap-2 rounded-xl border border-dashed px-4 py-6 cursor-pointer transition-all duration-200"
          style={{
            borderColor: dragOver ? "var(--accent)" : "var(--border-hover)",
            background: dragOver
              ? "var(--accent-muted)"
              : "var(--bg-elevated)",
          }}
        >
          {state.isUploading ? (
            <Loader2
              size={20}
              className="animate-spin"
              style={{ color: "var(--accent)" }}
            />
          ) : (
            <Upload size={20} style={{ color: "var(--fg-tertiary)" }} />
          )}
          <span
            className="text-xs text-center"
            style={{ color: "var(--fg-secondary)" }}
          >
            {state.isUploading
              ? "Processing..."
              : "Drop PDFs here or click to upload"}
          </span>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf"
          multiple
          className="hidden"
          onChange={handleFileChange}
        />
        {state.uploadError && (
          <p
            className="text-xs mt-2 px-1"
            style={{ color: "var(--error)" }}
          >
            {state.uploadError}
          </p>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-2 pb-2">
        {state.documents.length === 0 && !state.isUploading && (
          <p
            className="text-xs px-2 py-4 text-center"
            style={{ color: "var(--fg-muted)" }}
          >
            No sources added yet
          </p>
        )}

        {state.documents.map((doc) => {
          const isActive = state.activeDocumentId === doc.document_id;
          return (
            <button
              key={doc.document_id}
              onClick={() => selectDocument(doc.document_id)}
              className="w-full flex items-center gap-2 px-3 py-2.5 rounded-lg text-left transition-colors duration-150 group mb-1"
              style={{
                background: isActive ? "var(--bg-overlay)" : "transparent",
              }}
            >
              <FileText
                size={14}
                className="shrink-0"
                style={{
                  color: isActive ? "var(--accent)" : "var(--fg-tertiary)",
                }}
              />
              <div className="flex-1 min-w-0">
                <p
                  className="text-xs truncate"
                  style={{
                    color: isActive
                      ? "var(--fg-primary)"
                      : "var(--fg-secondary)",
                  }}
                >
                  {doc.filename}
                </p>
                <p
                  className="text-xs mt-0.5"
                  style={{
                    color: "var(--fg-muted)",
                    fontSize: 10,
                  }}
                >
                  {doc.num_chunks} chunks
                </p>
              </div>
              <button
                onClick={(e) => removeDocument(e, doc.document_id)}
                className="opacity-0 group-hover:opacity-100 p-1 rounded transition-opacity"
                style={{ color: "var(--fg-tertiary)" }}
              >
                <X size={12} />
              </button>
              {isActive && (
                <ChevronRight
                  size={12}
                  style={{ color: "var(--accent)" }}
                />
              )}
            </button>
          );
        })}
      </div>
    </aside>
  );
}
