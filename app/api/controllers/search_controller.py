from fastapi import APIRouter, HTTPException

from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from . import service_registry as svc

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search_document(request: SearchRequest):
    try:
        vector_store = svc.get_or_create_vector_store(request.document_id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store: {str(e)}")

    try:
        if request.search_method == "hybrid":
            results = vector_store.search(request.query, top_k=request.top_k)
        elif request.search_method == "faiss":
            results = vector_store.search_faiss_only(request.query, top_k=request.top_k)
        elif request.search_method == "bm25":
            results = vector_store.search_bm25_only(request.query, top_k=request.top_k)
        else:
            raise HTTPException(status_code=400, detail="Invalid search method. Use 'hybrid', 'faiss', or 'bm25'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

    result_models = [
        SearchResult(
            chunk_id=r['chunk_id'], topic=r['topic'], page=r['page'],
            text=r['text'], score=r['score'], score_breakdown=r.get('score_breakdown')
        )
        for r in results
    ]

    return SearchResponse(
        document_id=request.document_id,
        query=request.query,
        results=result_models,
        search_method=request.search_method,
        total_results=len(result_models)
    )
