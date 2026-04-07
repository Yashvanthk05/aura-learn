import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertCircle,
  AudioLines,
  CheckCircle2,
  Clock3,
  FileText,
  Loader2,
  MessageSquare,
  Plus,
  Upload,
  Video,
  X,
} from "lucide-react";
import { useApp } from "../store/AppContext";
import * as api from "../api/client";

const MAX_FILES = 10;
const DOC_EXTENSIONS = new Set([".pdf", ".pptx", ".docx", ".txt", ".md", ".csv"]);
const MEDIA_EXTENSIONS = new Set([".mp3", ".wav", ".mp4", ".m4a", ".mpeg", ".mpga", ".webm"]);
const ACCEPT_ATTR = ".pdf,.pptx,.docx,.txt,.md,.csv,.mp3,.wav,.mp4,.m4a,.mpeg,.mpga,.webm,audio/*,video/*";

const getExt = (name = "") => {
  const parts = name.toLowerCase().split(".");
  if (parts.length < 2) return "";
  return `.${parts.pop()}`;
};

const isMediaFile = (file) => {
  const ext = getExt(file.name);
  return file.type.startsWith("audio/") || file.type.startsWith("video/") || MEDIA_EXTENSIONS.has(ext);
};

const isSupportedFile = (file) => {
  const ext = getExt(file.name);
  return DOC_EXTENSIONS.has(ext) || MEDIA_EXTENSIONS.has(ext);
};

const formatUpdated = (value) => {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";

  const now = new Date();
  const isToday = date.toDateString() === now.toDateString();
  if (isToday) {
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  return date.toLocaleDateString([], { month: "short", day: "numeric" });
};

const sourceIcon = (file) => {
  if (isMediaFile(file)) {
    return file.type.startsWith("video/") || [".mp4", ".webm", ".mpeg"].includes(getExt(file.name)) ? Video : AudioLines;
  }
  return FileText;
};

export default function SourcesPanel({ width = 280, onResizeStart }) {
  const { state, dispatch } = useApp();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [chatTitle, setChatTitle] = useState("");
  const [pendingFiles, setPendingFiles] = useState([]);
  const [modalError, setModalError] = useState("");
  const fileInputRef = useRef(null);

  const selectChat = useCallback(
    async (chat) => {
      if (!chat) return;
      if (state.activeChatId === chat.session_id) return;

      dispatch({ type: "SET_ACTIVE_CHAT", payload: chat });
      dispatch({ type: "SET_CHAT_LOADING", payload: true });

      try {
        const [historyRes, sourcesRes, sessionRes] = await Promise.all([
          api.getChatHistory(chat.session_id).catch(() => ({ messages: [] })),
          api.getSessionSources(chat.session_id).catch(() => ({ sources: [] })),
          api.getSessionInfo(chat.session_id).catch(() => null),
        ]);

        dispatch({ type: "SET_MESSAGES", payload: historyRes.messages || [] });
        dispatch({ type: "SET_SOURCES", payload: sourcesRes.sources || [] });

        if (sessionRes) {
          dispatch({
            type: "UPDATE_CHAT_SESSION",
            payload: {
              sessionId: chat.session_id,
              patch: {
                updated_at: sessionRes.updated_at,
                message_count: sessionRes.message_count,
                title: sessionRes.title || chat.title || "New Chat",
                source_count: sessionRes.source_count ?? (sourcesRes.sources || []).length,
                metadata: sessionRes.metadata || chat.metadata || {},
              },
            },
          });
        }

        dispatch({ type: "SET_CHAT_LOADING", payload: false });
      } catch (err) {
        dispatch({ type: "SET_CHAT_ERROR", payload: err.message });
      }
    },
    [dispatch, state.activeChatId],
  );

  useEffect(() => {
    if (!state.user) return;

    api
      .listChatSessions()
      .then((res) => {
        dispatch({ type: "SET_CHAT_SESSIONS", payload: res.sessions || [] });
      })
      .catch(() => {
        dispatch({ type: "SET_CHAT_SESSIONS", payload: [] });
      });
  }, [dispatch, state.user]);

  useEffect(() => {
    if (state.activeChatId || state.chats.length === 0) return;
    void selectChat(state.chats[0]);
  }, [selectChat, state.activeChatId, state.chats]);

  const openModal = () => {
    setIsModalOpen(true);
    setChatTitle("");
    setPendingFiles([]);
    setModalError("");
  };

  const closeModal = () => {
    if (state.isCreatingChat || state.isUploading) return;
    setIsModalOpen(false);
    setModalError("");
  };

  const addFiles = useCallback((fileList) => {
    setPendingFiles((prev) => {
      const next = [...prev];
      const existingKeys = new Set(prev.map((item) => item.key));
      let firstError = "";

      for (const file of Array.from(fileList || [])) {
        const fileKey = `${file.name}-${file.size}-${file.lastModified}`;

        if (existingKeys.has(fileKey)) continue;

        if (!isSupportedFile(file)) {
          if (!firstError) {
            firstError = `Unsupported file: ${file.name}`;
          }
          continue;
        }

        if (next.length >= MAX_FILES) {
          if (!firstError) {
            firstError = `Maximum ${MAX_FILES} files per chat`;
          }
          break;
        }

        next.push({
          id: crypto.randomUUID(),
          key: fileKey,
          file,
          progress: 0,
          statusText: "Queued",
          stage: "queued",
          error: "",
        });
        existingKeys.add(fileKey);
      }

      if (firstError) setModalError(firstError);
      else setModalError("");

      return next;
    });
  }, []);

  const onFileInputChange = (e) => {
    addFiles(e.target.files);
    e.target.value = "";
  };

  const onDrop = (e) => {
    e.preventDefault();
    addFiles(e.dataTransfer.files);
  };

  const removePendingFile = (id) => {
    setPendingFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const updatePendingFile = (id, patch) => {
    setPendingFiles((prev) =>
      prev.map((item) => (item.id === id ? { ...item, ...patch } : item)),
    );
  };

  const ingestFile = useCallback(async (sessionId, item) => {
    const statusText = isMediaFile(item.file) ? "Transcribing..." : "Processing...";
    updatePendingFile(item.id, {
      stage: "uploading",
      statusText,
      progress: 8,
      error: "",
    });

    let fakeProgress = 8;
    const interval = setInterval(() => {
      fakeProgress = Math.min(fakeProgress + 4, 92);
      setPendingFiles((prev) =>
        prev.map((row) =>
          row.id === item.id
            ? { ...row, progress: Math.max(row.progress, fakeProgress) }
            : row,
        ),
      );
    }, 320);

    try {
      const result = await api.uploadSourceToSession(sessionId, item.file, {
        onProgress: (pct) => {
          const uploadMapped = Math.max(8, Math.min(75, Math.round((pct / 100) * 75)));
          setPendingFiles((prev) =>
            prev.map((row) =>
              row.id === item.id
                ? { ...row, progress: Math.max(row.progress, uploadMapped) }
                : row,
            ),
          );
        },
      });

      clearInterval(interval);
      updatePendingFile(item.id, {
        stage: "done",
        statusText: "Ready",
        progress: 100,
        error: "",
      });
      return { ok: true, result };
    } catch (err) {
      clearInterval(interval);
      updatePendingFile(item.id, {
        stage: "error",
        statusText: "Failed",
        error: err.message,
      });
      return { ok: false, error: err.message };
    }
  }, []);

  const createChat = useCallback(async () => {
    if (pendingFiles.length === 0) {
      setModalError("Add at least one file before creating a chat.");
      return;
    }

    dispatch({ type: "SET_CREATING_CHAT", payload: true });
    dispatch({ type: "SET_UPLOADING", payload: true });
    setModalError("");

    try {
      const created = await api.createWorkspaceChat(chatTitle.trim() || undefined);
      const newChat = {
        session_id: created.session_id,
        document_id: created.document_id,
        title: created.title || chatTitle.trim() || "New Chat",
        source_count: 0,
        message_count: 0,
        created_at: created.created_at,
        updated_at: created.created_at,
        metadata: {
          title: created.title || chatTitle.trim() || "New Chat",
          sources: [],
        },
      };

      dispatch({ type: "ADD_CHAT_SESSION", payload: newChat });
      dispatch({ type: "SET_ACTIVE_CHAT", payload: newChat });
      dispatch({ type: "SET_MESSAGES", payload: [] });
      dispatch({ type: "SET_SOURCES", payload: [] });

      const uploads = await Promise.all(
        pendingFiles.map((item) => ingestFile(created.session_id, item)),
      );
      const successful = uploads.filter((r) => r.ok).length;

      if (successful === 0) {
        setModalError("Chat created, but all file ingestions failed. Please try again.");
      } else {
        const [sourcesRes, sessionRes] = await Promise.all([
          api.getSessionSources(created.session_id),
          api.getSessionInfo(created.session_id),
        ]);

        const sources = sourcesRes.sources || [];
        dispatch({ type: "SET_SOURCES", payload: sources });
        dispatch({
          type: "UPDATE_CHAT_SESSION",
          payload: {
            sessionId: created.session_id,
            patch: {
              source_count: sources.length,
              updated_at: sessionRes.updated_at,
              title: sessionRes.title || newChat.title,
              metadata: sessionRes.metadata || newChat.metadata,
            },
          },
        });

        setIsModalOpen(false);
        setChatTitle("");
        setPendingFiles([]);
        setModalError("");
      }
    } catch (err) {
      setModalError(err.message);
      dispatch({ type: "SET_CREATE_CHAT_ERROR", payload: err.message });
    } finally {
      dispatch({ type: "SET_CREATING_CHAT", payload: false });
      dispatch({ type: "SET_UPLOADING", payload: false });
    }
  }, [chatTitle, dispatch, ingestFile, pendingFiles]);

  return (
    <>
      <aside
        className='flex flex-col h-full border-r'
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
          aria-label='Resize chats panel'
          aria-orientation='vertical'
          onMouseDown={(e) => {
            e.preventDefault();
            onResizeStart?.();
          }}
          className='absolute top-0 right-0 h-full w-1.5 cursor-col-resize'
          style={{ transform: "translateX(50%)", zIndex: 20 }}
        />

        <div
          className='flex items-center justify-between gap-2 px-4 py-3 border-b'
          style={{ borderColor: "var(--border)" }}
        >
          <div className='flex items-center gap-2'>
            <div
              className='w-7 h-7 rounded-lg flex items-center justify-center'
              style={{ background: "var(--accent-muted)" }}
            >
              <MessageSquare size={14} style={{ color: "var(--accent)" }} />
            </div>
            <span className='font-medium text-sm' style={{ color: "var(--fg-primary)" }}>
              Chats
            </span>
          </div>

          <button
            onClick={openModal}
            className='inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-colors'
            style={{
              background: "var(--accent-muted)",
              color: "var(--accent)",
              border: "1px solid var(--border)",
            }}
          >
            <Plus size={12} />
            New Chat
          </button>
        </div>

        <div className='px-2 py-2 border-b' style={{ borderColor: "var(--border)" }}>
          <p className='text-[10px] px-2' style={{ color: "var(--fg-muted)" }}>
            One chat can include up to {MAX_FILES} source files.
          </p>
        </div>

        <div className='flex-1 overflow-y-auto p-2'>
          {state.chats.length === 0 ? (
            <div className='px-3 py-5 rounded-lg text-center' style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
              <p className='text-xs mb-1' style={{ color: "var(--fg-secondary)" }}>
                No chats yet
              </p>
              <p className='text-[10px]' style={{ color: "var(--fg-muted)" }}>
                Create your first chat and upload sources to start.
              </p>
            </div>
          ) : (
            state.chats.map((chat) => {
              const isActive = state.activeChatId === chat.session_id;
              return (
                <button
                  key={chat.session_id}
                  onClick={() => selectChat(chat)}
                  className='w-full text-left rounded-xl px-3 py-2.5 mb-1.5 transition-colors'
                  style={{
                    background: isActive ? "var(--bg-overlay)" : "transparent",
                    border: `1px solid ${isActive ? "var(--border-active)" : "transparent"}`,
                  }}
                >
                  <div className='flex items-start justify-between gap-2'>
                    <p className='text-xs font-medium truncate' style={{ color: isActive ? "var(--fg-primary)" : "var(--fg-secondary)" }}>
                      {chat.title || "New Chat"}
                    </p>
                    <span className='text-[10px] shrink-0' style={{ color: "var(--fg-muted)" }}>
                      {formatUpdated(chat.updated_at)}
                    </span>
                  </div>
                  <div className='flex items-center gap-2 mt-1'>
                    <span className='text-[10px] px-1.5 py-0.5 rounded' style={{ background: "var(--bg-elevated)", color: "var(--fg-tertiary)" }}>
                      {chat.source_count ?? 0} sources
                    </span>
                    <span className='text-[10px] px-1.5 py-0.5 rounded' style={{ background: "var(--bg-elevated)", color: "var(--fg-tertiary)" }}>
                      {chat.message_count ?? 0} msgs
                    </span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </aside>

      {isModalOpen && (
        <div
          className='fixed inset-0 z-50 flex items-center justify-center px-4'
          style={{ background: "rgba(10,10,11,0.64)" }}
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) closeModal();
          }}
        >
          <div
            className='w-full max-w-2xl rounded-2xl border shadow-2xl overflow-hidden'
            style={{
              background: "var(--bg-surface)",
              borderColor: "var(--border)",
            }}
          >
            <div className='flex items-center justify-between px-5 py-4 border-b' style={{ borderColor: "var(--border)" }}>
              <div>
                <h3 className='text-sm font-semibold' style={{ color: "var(--fg-primary)" }}>
                  Create New Chat
                </h3>
                <p className='text-xs mt-0.5' style={{ color: "var(--fg-muted)" }}>
                  Upload documents and media into one shared knowledge base.
                </p>
              </div>
              <button
                onClick={closeModal}
                disabled={state.isCreatingChat || state.isUploading}
                className='w-8 h-8 rounded-lg flex items-center justify-center'
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--fg-secondary)",
                  border: "1px solid var(--border)",
                  opacity: state.isCreatingChat || state.isUploading ? 0.5 : 1,
                }}
              >
                <X size={14} />
              </button>
            </div>

            <div className='px-5 py-4 space-y-4'>
              <div>
                <label className='text-xs mb-1.5 block' style={{ color: "var(--fg-secondary)" }}>
                  Chat title (optional)
                </label>
                <input
                  type='text'
                  value={chatTitle}
                  onChange={(e) => setChatTitle(e.target.value)}
                  placeholder='My Research Notebook'
                  className='w-full rounded-lg px-3 py-2 text-sm outline-none'
                  style={{
                    background: "var(--bg-elevated)",
                    color: "var(--fg-primary)",
                    border: "1px solid var(--border)",
                  }}
                />
              </div>

              <div
                role='button'
                tabIndex={0}
                onDragOver={(e) => e.preventDefault()}
                onDrop={onDrop}
                onClick={() => fileInputRef.current?.click()}
                onKeyDown={(e) => e.key === "Enter" && fileInputRef.current?.click()}
                className='rounded-xl border border-dashed px-4 py-8 cursor-pointer text-center transition-colors'
                style={{
                  borderColor: "var(--border-hover)",
                  background: "var(--bg-elevated)",
                }}
              >
                <Upload size={22} className='mx-auto mb-2' style={{ color: "var(--accent)" }} />
                <p className='text-xs' style={{ color: "var(--fg-secondary)" }}>
                  Drop files here or click to browse
                </p>
                <p className='text-[10px] mt-1' style={{ color: "var(--fg-muted)" }}>
                  PDF, PPTX, DOCX, TXT, MD, CSV, MP3, WAV, MP4, M4A, WEBM
                </p>
                <p className='text-[10px] mt-1' style={{ color: "var(--fg-muted)" }}>
                  Up to {MAX_FILES} files per chat
                </p>
              </div>

              <input
                ref={fileInputRef}
                type='file'
                multiple
                accept={ACCEPT_ATTR}
                className='hidden'
                onChange={onFileInputChange}
              />

              {pendingFiles.length > 0 && (
                <div className='rounded-xl p-3 space-y-2 max-h-64 overflow-y-auto' style={{ background: "var(--bg-elevated)", border: "1px solid var(--border)" }}>
                  {pendingFiles.map((item) => {
                    const Icon = sourceIcon(item.file);
                    const isBusy = item.stage === "uploading";
                    return (
                      <div key={item.id} className='rounded-lg px-3 py-2' style={{ background: "var(--bg-surface)", border: "1px solid var(--border)" }}>
                        <div className='flex items-center gap-2'>
                          <Icon size={14} style={{ color: "var(--fg-tertiary)" }} />
                          <span className='text-xs truncate flex-1' style={{ color: "var(--fg-primary)" }}>
                            {item.file.name}
                          </span>
                          {item.stage === "done" && <CheckCircle2 size={14} style={{ color: "var(--success)" }} />}
                          {item.stage === "error" && <AlertCircle size={14} style={{ color: "var(--error)" }} />}
                          {!isBusy && item.stage !== "done" && (
                            <button
                              onClick={() => removePendingFile(item.id)}
                              className='w-5 h-5 rounded flex items-center justify-center'
                              style={{ color: "var(--fg-muted)" }}
                            >
                              <X size={12} />
                            </button>
                          )}
                        </div>

                        <div className='mt-1.5 h-1.5 rounded-full overflow-hidden' style={{ background: "var(--bg-overlay)" }}>
                          <div
                            className='h-full rounded-full transition-[width] duration-200'
                            style={{
                              width: `${item.progress}%`,
                              background:
                                item.stage === "error"
                                  ? "var(--error)"
                                  : item.stage === "done"
                                    ? "var(--success)"
                                    : "var(--accent)",
                            }}
                          />
                        </div>

                        <div className='mt-1 flex items-center justify-between'>
                          <span className='text-[10px]' style={{ color: "var(--fg-muted)" }}>
                            {item.error || item.statusText}
                          </span>
                          <span className='text-[10px]' style={{ color: "var(--fg-muted)", fontFamily: "var(--font-mono)" }}>
                            {Math.round(item.progress)}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              {(modalError || state.createChatError) && (
                <p className='text-xs px-1' style={{ color: "var(--error)" }}>
                  {modalError || state.createChatError}
                </p>
              )}
            </div>

            <div className='flex items-center justify-between gap-2 px-5 py-4 border-t' style={{ borderColor: "var(--border)" }}>
              <p className='text-[11px] flex items-center gap-1.5' style={{ color: "var(--fg-muted)" }}>
                <Clock3 size={12} />
                Media files are transcribed automatically during ingestion.
              </p>

              <button
                onClick={createChat}
                disabled={state.isCreatingChat || state.isUploading || pendingFiles.length === 0}
                className='inline-flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-colors'
                style={{
                  background: "var(--accent)",
                  color: "#fff",
                  opacity:
                    state.isCreatingChat || state.isUploading || pendingFiles.length === 0
                      ? 0.6
                      : 1,
                }}
              >
                {state.isCreatingChat || state.isUploading ? (
                  <>
                    <Loader2 size={13} className='animate-spin' />
                    Creating...
                  </>
                ) : (
                  "Create Chat"
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
