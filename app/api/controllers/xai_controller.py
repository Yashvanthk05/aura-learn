from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    ExplainExtractiveRequest, ExplainExtractiveResponse, SentenceExplanation,
    ExplainSearchRequest, ExplainSearchResponse, ExplainedSearchResult,
    ExplainAbstractiveRequest, ExplainAbstractiveResponse,
    SentenceContribution, TokenConfidence,
)
from app.services.xai_service import (
    ExplainableExtractiveService,
    ExplainableSearchService,
    ExplainableAbstractiveService,
)
from . import service_registry as svc

router = APIRouter()


def _get_combined_text(document_id: str, chunk_ids=None) -> str:
    chunks = svc.document_manager.get_chunks(document_id, chunk_ids)
    if chunks is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks found for specified IDs")
    return " ".join([chunk["text"] for chunk in chunks])


@router.post("/explain/extractive", response_model=ExplainExtractiveResponse)
async def explain_extractive(request: ExplainExtractiveRequest):
    """
    Explain extractive summarization decisions.

    Returns per-sentence importance scores, attention weights,
    and sensitivity analysis showing why each sentence was selected or rejected.
    """
    if svc.extractive_summarizer is None:
        raise HTTPException(status_code=503, detail="Extractive summarizer not available")

    combined_text = _get_combined_text(request.document_id, request.chunk_ids)

    try:
        xai = ExplainableExtractiveService(svc.extractive_summarizer)
        result = xai.explain_extractive(combined_text, request.num_sentences)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return ExplainExtractiveResponse(
            summary=result["summary"],
            num_sentences_input=result["num_sentences_input"],
            num_sentences_selected=result["num_sentences_selected"],
            selected_indices=result["selected_indices"],
            average_score_selected=result["average_score_selected"],
            average_score_all=result["average_score_all"],
            score_distribution=result["score_distribution"],
            sentences=[SentenceExplanation(**s) for s in result["sentences"]],
            explanation_methods=result["explanation_methods"],
            xai_type=result["xai_type"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Explanation generation failed: {str(e)}"
        )


@router.post("/explain/search", response_model=ExplainSearchResponse)
async def explain_search(request: ExplainSearchRequest):
    """
    Explain hybrid search results.

    Shows the scoring breakdown (FAISS, BM25, TF-IDF) for each result
    and provides human-readable explanations for ranking decisions.
    """
    try:
        vector_store = svc.get_or_create_vector_store(request.document_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to load vector store: {str(e)}"
        )

    try:
        results = vector_store.search(request.query, top_k=request.top_k)

        xai = ExplainableSearchService()
        explanation = xai.explain_search(request.query, results, request.top_k)

        return ExplainSearchResponse(
            query=explanation["query"],
            total_results_explained=explanation["total_results_explained"],
            results=[ExplainedSearchResult(**r) for r in explanation["results"]],
            scoring_weights=explanation["scoring_weights"],
            xai_type=explanation["xai_type"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Search explanation failed: {str(e)}"
        )


@router.post("/explain/abstractive", response_model=ExplainAbstractiveResponse)
async def explain_abstractive(request: ExplainAbstractiveRequest):
    """
    Explain abstractive summarization decisions.

    Uses leave-one-out sentence attribution to show which input sentences
    most influenced the generated summary, plus per-token confidence scores.
    """
    if svc.abstractive_summarizer is None:
        raise HTTPException(
            status_code=503, detail="Abstractive summarizer not available"
        )

    combined_text = _get_combined_text(request.document_id, request.chunk_ids)

    try:
        xai = ExplainableAbstractiveService(svc.abstractive_summarizer)
        result = xai.explain_abstractive(
            combined_text,
            max_length=request.max_length,
            min_length=request.min_length,
            generate_shap=request.generate_shap,
        )

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        from app.models.schemas import ShapExplanation
        shap_explanation = None
        if result.get("shap_explanation"):
            shap_explanation = ShapExplanation(**result["shap_explanation"])

        return ExplainAbstractiveResponse(
            original_text_sentences=result["original_text_sentences"],
            summary=result["summary"],
            summary_word_count=result["summary_word_count"],
            compression_ratio=result["compression_ratio"],
            sentence_contributions=[
                SentenceContribution(**c) for c in result["sentence_contributions"]
            ],
            most_influential_sentence=(
                SentenceContribution(**result["most_influential_sentence"])
                if result["most_influential_sentence"]
                else None
            ),
            token_confidence=[
                TokenConfidence(**t) for t in result["token_confidence"]
            ],
            shap_explanation=shap_explanation,
            explanation_methods=result["explanation_methods"],
            xai_type=result["xai_type"],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Abstractive explanation failed: {str(e)}",
        )


@router.get("/explain/methods")
async def list_xai_methods():
    """List all available XAI methods and their descriptions."""
    return {
        "available_methods": [
            {
                "endpoint": "/api/v1/explain/extractive",
                "method": "POST",
                "xai_type": "Post-hoc + Deep Explanation",
                "techniques": [
                    "Sentence importance scoring",
                    "Attention weight extraction",
                    "Leave-one-out sensitivity analysis",
                ],
                "model": "BiLSTM Extractive Summarizer",
                "description": "Explains why specific sentences were selected for the extractive summary.",
            },
            {
                "endpoint": "/api/v1/explain/search",
                "method": "POST",
                "xai_type": "Transparent Approximation (BETA)",
                "techniques": [
                    "Score decomposition (FAISS + BM25 + TF-IDF)",
                    "Query-result word overlap analysis",
                    "Dominant method identification",
                ],
                "model": "Hybrid Vector Search",
                "description": "Explains why each search result was ranked in its position.",
            },
            {
                "endpoint": "/api/v1/explain/abstractive",
                "method": "POST",
                "xai_type": "Post-hoc Sensitivity Analysis",
                "techniques": [
                    "Leave-one-out sentence attribution",
                    "Per-token generation confidence",
                    "Compression ratio analysis",
                ],
                "model": "T5 Abstractive Summarizer",
                "description": "Explains which input sentences influenced the generated summary most.",
            },
        ],
        "syllabus_coverage": {
            "Module 2 - Interpretability": "Sentence scoring, feature importance",
            "Module 3 - Deep Explanation": "Attention weights extraction",
            "Module 4 - XAI Models": "Post-hoc (PHE), Transparent Approximation (BETA)",
            "Module 5 - XAI Methods": "Sensitivity analysis, feature attribution (SHAP-like)",
            "Module 6 - Trust": "Score decomposition, confidence metrics",
        },
    }