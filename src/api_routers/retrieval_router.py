# src/api_routers/retrieval_router.py
# Router for API endpoints related to data retrieval: search, rerank, and fetching chunk details.

from fastapi import APIRouter, HTTPException, Body
from typing import List, Dict, Optional, Any, Tuple

# Import shared app_state, Pydantic models, and other necessary modules
from ..main_api import app_state # Access global app state
from ..api_models import (
    QueryRequest, RerankRequest, SearchResultItem, SearchResponse, ChunkDetailResponse
)
from .. import embedding_module # For generating query embeddings
from .. import ragdoll_config   # For default batch sizes etc.

router = APIRouter()

@router.post("/search", response_model=SearchResponse, summary="Search for Relevant Chunks")
async def endpoint_search_chunks(request: QueryRequest):
    """
    Performs a vector search for the given query.
    Embeds the query, queries the loaded vector store, and returns ranked chunks,
    hydrating them with text and metadata if available.
    """
    if not app_state.vector_store_manager: 
        raise HTTPException(status_code=503, detail="Vector store not loaded. Cannot perform search.")
    if not app_state.query_embedding_model_name or \
       not app_state.query_embedding_model_type or \
       app_state.query_embedding_model_dim is None: 
        raise HTTPException(status_code=503, detail="Query embedding model parameters are not configured.")
    
    query_vector_np, _ = embedding_module.generate_embeddings( 
        [request.query], 
        app_state.query_embedding_model_name, 
        app_state.query_embedding_model_type, 
        app_state.query_embedding_model_dim, 
        app_state.query_embedding_model_class, 
        device=app_state.embedding_device
    )
    if query_vector_np.shape[0] == 0: 
        raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
    
    retrieved_items = app_state.vector_store_manager.query_store(
        query_vector_np[0], k=request.top_k, threshold=request.threshold
    )
    
    search_results: List[SearchResultItem] = []
    if app_state.supporting_data["text_chunks"] and \
       app_state.supporting_data["chunk_ids"] and \
       app_state.supporting_data["detailed_metadata"]:
        id_to_idx = {id_val: idx for idx, id_val in enumerate(app_state.supporting_data["chunk_ids"])}
        for item_id, score in retrieved_items:
            idx = id_to_idx.get(item_id)
            if idx is not None and \
               idx < len(app_state.supporting_data["text_chunks"]) and \
               idx < len(app_state.supporting_data["detailed_metadata"]):
                search_results.append(SearchResultItem(
                    id=item_id, 
                    text=app_state.supporting_data["text_chunks"][idx], 
                    score=score, 
                    metadata=app_state.supporting_data["detailed_metadata"][idx]
                ))
            else: 
                search_results.append(SearchResultItem(id=item_id, score=score, metadata={"error": "Chunk data missing for ID", "id_queried": item_id}))
    else: 
        search_results = [SearchResultItem(id=item_id, score=score, metadata={"warning": "Supporting data not fully loaded"}) for item_id, score in retrieved_items]
        
    return SearchResponse(query=request.query, results=search_results)

@router.post("/rerank", response_model=List[SearchResultItem], summary="Rerank Provided Chunks")
async def endpoint_rerank_chunks(request: RerankRequest):
    """
    Reranks a list of provided chunks based on relevance to a query using a cross-encoder model.
    """
    if not app_state.reranker_instance or not app_state.reranker_instance.model: 
        raise HTTPException(status_code=503, detail="Reranker model not loaded.")
    
    chunks_for_reranker: List[Tuple[str, str]] = []
    original_chunk_data_map: Dict[str, Dict[str, Any]] = {} 

    for chunk_item in request.retrieved_chunks: # RerankRequest.retrieved_chunks is List[RerankRequestItem]
        if chunk_item.id and chunk_item.text is not None:
            chunks_for_reranker.append((chunk_item.id, chunk_item.text))
            original_chunk_data_map[chunk_item.id] = chunk_item.model_dump() # Store original Pydantic model as dict
    
    if not chunks_for_reranker: return []
    
    reranked_ids_scores: List[Tuple[str, float]] = app_state.reranker_instance.rerank(
        request.query, chunks_for_reranker, request.top_n_rerank, 
        batch_size=ragdoll_config.DEFAULT_RERANKER_BATCH_SIZE
    )
    
    response_items: List[SearchResultItem] = []
    for item_id, new_score in reranked_ids_scores:
        original_data = original_chunk_data_map.get(item_id)
        if original_data:
            response_items.append(SearchResultItem(
                id=item_id, 
                text=original_data.get("text"), 
                score=new_score, 
                metadata=original_data.get("metadata")
            ))
    return response_items

@router.get("/get_chunk_details/{chunk_id}", response_model=ChunkDetailResponse, summary="Get Full Details for a Chunk")
async def endpoint_get_chunk_details(chunk_id: str):
    """
    Retrieves the full text content and all associated metadata for a specific chunk ID.
    """
    if not all(app_state.supporting_data.get(key) for key in ["chunk_ids", "text_chunks", "detailed_metadata"]): 
        raise HTTPException(status_code=503, detail="Supporting chunk data not loaded.")
    try:
        idx = app_state.supporting_data["chunk_ids"].index(chunk_id)
        return ChunkDetailResponse(
            id=chunk_id, 
            text=app_state.supporting_data["text_chunks"][idx], 
            metadata=app_state.supporting_data["detailed_metadata"][idx]
        )
    except ValueError: 
        raise HTTPException(status_code=404, detail=f"Chunk ID '{chunk_id}' not found.")
    except IndexError: 
        raise HTTPException(status_code=500, detail=f"Data inconsistency for chunk ID '{chunk_id}'.")