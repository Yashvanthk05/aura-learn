const BASE = "/api/v1";

const getHeaders = (customHeaders = {}) => {
  const token = localStorage.getItem("access_token");
  const headers = { "Content-Type": "application/json", ...customHeaders };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }
  return headers;
};

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: getHeaders(options.headers),
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Request failed");
  }
  return res.json();
}

export const getHealth = () => request("/health");

export function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);

  const token = localStorage.getItem("access_token");
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  return fetch(`${BASE}/upload`, { method: "POST", body: form, headers }).then(
    async (r) => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || "Upload failed");
      }
      return r.json();
    },
  );
}

export const listDocuments = () => request("/documents");

export const getDocument = (docId) => request(`/document/${docId}`);

export const deleteDocument = (docId) =>
  request(`/document/${docId}`, { method: "DELETE" });

export const createChatSession = (documentId) =>
  request("/chat/session", {
    method: "POST",
    body: JSON.stringify({ document_id: documentId }),
  });

export const createWorkspaceChat = (title) =>
  request("/chat/workspace", {
    method: "POST",
    body: JSON.stringify({ title }),
  });

export const listChatSessions = () => request("/chat/sessions");

export const getSessionInfo = (sessionId) => request(`/chat/session/${sessionId}`);

export const getSessionSources = (sessionId) =>
  request(`/chat/session/${sessionId}/sources`);

export function uploadSourceToSession(sessionId, file, opts = {}) {
  const form = new FormData();
  form.append("file", file);

  const token = localStorage.getItem("access_token");

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/chat/session/${sessionId}/sources`, true);

    if (token) {
      xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    }

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && typeof opts.onProgress === "function") {
        const pct = Math.round((event.loaded / event.total) * 100);
        opts.onProgress(pct);
      }
    };

    xhr.onerror = () => {
      reject(new Error("Network error during upload"));
    };

    xhr.onreadystatechange = () => {
      if (xhr.readyState !== XMLHttpRequest.DONE) return;

      let payload = null;
      if (xhr.responseText) {
        try {
          payload = JSON.parse(xhr.responseText);
        } catch {
          payload = null;
        }
      }

      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(payload);
        return;
      }

      reject(new Error(payload?.detail || "Upload failed"));
    };

    xhr.send(form);
  });
}

export async function getSourceFileBlob(sessionId, sourceId) {
  const token = localStorage.getItem("access_token");
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${BASE}/chat/session/${sessionId}/source/${sourceId}`, {
    method: "GET",
    headers,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || "Failed to fetch source file");
  }

  const blob = await res.blob();
  const contentType = res.headers.get("Content-Type") || "application/octet-stream";
  const disposition = res.headers.get("Content-Disposition") || "";
  const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);

  return {
    blob,
    contentType,
    filename: filenameMatch?.[1] || `source-${sourceId}`,
  };
}

export const chatQuery = (sessionId, query, opts = {}) =>
  request("/chat/query", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      query,
      use_hybrid_search: opts.useHybridSearch ?? true,
      top_k: opts.topK ?? 5,
      include_history: opts.includeHistory ?? true,
      max_context_chunks: opts.maxContextChunks ?? 5,
      model_type: opts.modelType ?? "extractive",
    }),
  });

export const getChatHistory = (sessionId, maxMessages) => {
  const qs = maxMessages ? `?max_messages=${maxMessages}` : "";
  return request(`/chat/session/${sessionId}/history${qs}`);
};

export const summarize = (documentId, opts = {}) =>
  request("/summarize", {
    method: "POST",
    body: JSON.stringify({
      document_id: documentId,
      summarization_type: opts.type ?? "extractive",
      num_sentences: opts.numSentences,
      max_length: opts.maxLength ?? 150,
      min_length: opts.minLength ?? 40,
      chunk_ids: opts.chunkIds,
    }),
  });

export const summarizeAndAudio = (documentId, opts = {}) => {
  const form = new FormData();
  form.append("document_id", documentId);
  form.append("summarization_type", opts.type ?? "abstractive");
  form.append("language", opts.language ?? "en");
  form.append("max_length", opts.maxLength ?? 150);
  form.append("min_length", opts.minLength ?? 40);

  if (opts.numSentences) form.append("num_sentences", opts.numSentences);
  if (opts.chunkIds) form.append("chunk_ids", JSON.stringify(opts.chunkIds));
  if (opts.speakerAudio) form.append("speaker_audio", opts.speakerAudio);

  const token = localStorage.getItem("access_token");
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  return fetch(`${BASE}/summarize-and-audio`, {
    method: "POST",
    body: form,
    headers,
  }).then(async (r) => {
    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(err.detail || "Request failed");
    }
    return r.json();
  });
};

export const generateAudiobook = (text, language = "en") =>
  request("/generate-audiobook", {
    method: "POST",
    body: JSON.stringify({ text, language }),
  });

export const searchDocument = (documentId, query, opts = {}) =>
  request("/search", {
    method: "POST",
    body: JSON.stringify({
      document_id: documentId,
      query,
      top_k: opts.topK ?? 5,
      search_method: opts.method ?? "hybrid",
    }),
  });

export const explainExtractive = (documentId, opts = {}) =>
  request("/explain/extractive", {
    method: "POST",
    body: JSON.stringify({
      document_id: documentId,
      num_sentences: opts.numSentences ?? 3,
      chunk_ids: opts.chunkIds,
      generate_lrp: opts.generateLrp ?? false,
    }),
  });

export const explainAbstractive = (documentId, opts = {}) =>
  request("/explain/abstractive", {
    method: "POST",
    body: JSON.stringify({
      document_id: documentId,
      max_length: opts.maxLength ?? 150,
      min_length: opts.minLength ?? 50,
      chunk_ids: opts.chunkIds,
      generate_shap: opts.generateShap ?? false,
    }),
  });

export const explainSearch = (documentId, query, opts = {}) =>
  request("/explain/search", {
    method: "POST",
    body: JSON.stringify({
      document_id: documentId,
      query,
      top_k: opts.topK ?? 5,
    }),
  });

export function transcribeMedia(file, opts = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("summarization_type", opts.type ?? "extractive");
  form.append("num_sentences", String(opts.numSentences ?? 3));
  form.append("max_length", String(opts.maxLength ?? 150));
  form.append("min_length", String(opts.minLength ?? 40));

  const token = localStorage.getItem("access_token");
  const headers = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;

  return fetch(`${BASE}/transcribe/`, { method: "POST", body: form, headers }).then(
    async (r) => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || "Transcription failed");
      }
      return r.json();
    },
  );
}

export const googleLogin = (token) =>
  request("/auth/google", {
    method: "POST",
    body: JSON.stringify({ token }),
  });
