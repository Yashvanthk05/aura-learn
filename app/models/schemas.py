from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: Dict[str, Any]


class UploadPDFResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    message: str


class ChunkInfo(BaseModel):
    chunk_id: Any
    topic: str
    page: Any
    text_preview: str


class DocumentInfoResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    chunks: List[ChunkInfo]


class SummarizeRequest(BaseModel):
    document_id: str
    chunk_ids: Optional[List[str]] = None
    summarization_type: str = "extractive"
    num_sentences: Optional[int] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None


class SummarizeResponse(BaseModel):
    document_id: str
    summarization_type: str
    summary: str
    num_chunks_processed: int
    metadata: dict


class SummarizeAndAudioRequest(BaseModel):
    document_id: str
    chunk_ids: Optional[List[str]] = None
    summarization_type: str = "extractive"
    num_sentences: Optional[int] = None
    max_length: Optional[int] = None
    min_length: Optional[int] = None
    language: str = "en"


class SummarizeAndAudioResponse(BaseModel):
    document_id: str
    summarization_type: str
    summary: str
    audio_url: str
    audio_filename: str
    num_chunks_processed: int


class AudiobookRequest(BaseModel):
    text: str
    language: str = "en"


class AudiobookResponse(BaseModel):
    audio_url: str
    filename: str
    text_length: int
    language: str
    message: str


class SearchRequest(BaseModel):
    document_id: str
    query: str
    top_k: int = 5
    search_method: str = "hybrid"


class SearchResult(BaseModel):
    chunk_id: Any
    topic: str
    page: Any
    text: str
    score: float
    score_breakdown: Optional[dict] = None


class SearchResponse(BaseModel):
    document_id: str
    query: str
    results: List[SearchResult]
    search_method: str
    total_results: int


class CreateSessionRequest(BaseModel):
    document_id: str
    metadata: Optional[dict] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    document_id: str
    created_at: str
    message: str


class Citation(BaseModel):
    id: int
    chunk_id: Any
    topic: str
    page: Any
    source: str
    text_snippet: str
    score: float
    relevance: str


class ChatQueryRequest(BaseModel):
    session_id: str
    query: str
    use_hybrid_search: bool = True
    top_k: int = 5
    include_history: bool = True
    max_context_chunks: int = 5


class ChatQueryResponse(BaseModel):
    session_id: str
    query: str
    response: str
    citations: List[Citation]
    retrieved_chunks: int
    search_method: str
    timestamp: str


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    citations: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


class ConversationHistoryResponse(BaseModel):
    session_id: str
    document_id: str
    messages: List[ChatMessage]
    message_count: int


class SessionInfoResponse(BaseModel):
    session_id: str
    document_id: str
    created_at: str
    updated_at: str
    message_count: int
    metadata: Optional[dict] = None


class SentenceExplanation(BaseModel):
    index: int
    text: str
    importance_score: float
    is_selected: bool
    sensitivity: float
    word_count: int
    most_attended_sentence: Optional[int] = None
    attention_to_others: Optional[List[float]] = None


class ExplainExtractiveRequest(BaseModel):
    document_id: str
    chunk_ids: Optional[List[str]] = None
    num_sentences: int = 3


class ExplainExtractiveResponse(BaseModel):
    summary: str
    num_sentences_input: int
    num_sentences_selected: int
    selected_indices: List[int]
    average_score_selected: float
    average_score_all: float
    score_distribution: dict
    sentences: List[SentenceExplanation]
    explanation_methods: List[str]
    xai_type: str


class ExplainSearchRequest(BaseModel):
    document_id: str
    query: str
    top_k: int = 5


class ExplainedSearchResult(BaseModel):
    rank: int
    chunk_id: Any
    topic: str
    page: Any
    text_preview: str
    combined_score: float
    score_breakdown: Dict[str, float]
    dominant_scoring_method: str
    query_word_overlap: List[str]
    word_overlap_ratio: float
    explanation: str


class ExplainSearchResponse(BaseModel):
    query: str
    total_results_explained: int
    results: List[ExplainedSearchResult]
    scoring_weights: dict
    xai_type: str


class ExplainAbstractiveRequest(BaseModel):
    document_id: str
    chunk_ids: Optional[List[str]] = None
    max_length: int = 150
    min_length: int = 50


class SentenceContribution(BaseModel):
    index: int
    text: str
    contribution_score: float
    without_summary: str
    normalized_contribution: float = 0.0


class TokenConfidence(BaseModel):
    token: str
    confidence: float
    is_high_confidence: bool


class ExplainAbstractiveResponse(BaseModel):
    original_text_sentences: List[str]
    summary: str
    summary_word_count: int
    compression_ratio: float
    sentence_contributions: List[SentenceContribution]
    most_influential_sentence: Optional[SentenceContribution] = None
    token_confidence: List[TokenConfidence]
    explanation_methods: List[str]
    xai_type: str
