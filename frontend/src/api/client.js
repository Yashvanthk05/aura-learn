const BASE = "/api/v1";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
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
  return fetch(`${BASE}/upload`, { method: "POST", body: form }).then(
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

  return fetch(`${BASE}/summarize-and-audio`, {
    method: "POST",
    body: form,
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
  return fetch(`${BASE}/transcribe/`, { method: "POST", body: form }).then(
    async (r) => {
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(err.detail || "Transcription failed");
      }
      return r.json();
    },
  );
}
