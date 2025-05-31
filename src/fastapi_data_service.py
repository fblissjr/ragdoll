from fastapi import FastAPI, HTTPException, Query, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import json
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
from datetime import datetime
import torch

import common_utils
import embedding_module
from vector_store_manager import VectorStoreManager
import reranker_module
import pipeline_orchestrator 

app = FastAPI(
    title="RAGdoll Data Processing & Retrieval Service",
    version="0.3.3" 
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AppState: # Same
    vector_store_manager: Optional[VectorStoreManager] = None
    reranker_instance: Optional[reranker_module.Reranker] = None
    supporting_data: Dict[str, Any] = {"text_chunks": None, "chunk_ids": None, "detailed_metadata": None}
    query_embedding_model_name: str = common_utils.DEFAULT_QUERY_EMBEDDING_MODEL
    query_embedding_model_type: Optional[str] = None; query_embedding_model_class: Optional[str] = None
    query_embedding_model_dim: Optional[int] = None; embedding_device: str = "cpu"
    pipeline_default_vector_data_dir: str = "vector_store_data"
    pipeline_reranker_model_name: str = common_utils.DEFAULT_RERANKER_MODEL
app_state = AppState()

class PipelineRunRequest(BaseModel): # Updated to match CLI and common_utils structure
    docs_folder: str = Field(default="docs_input")
    vector_data_dir: str = Field(default=app_state.pipeline_default_vector_data_dir)
    overwrite: bool = Field(default=True)
    gpu_device_id: int = Field(default=common_utils.DEFAULT_GPU_DEVICE_ID_PROCESSING)
    verbose: bool = Field(default=False)

    embedding_model_name: str = Field(default=common_utils.DEFAULT_PIPELINE_EMBEDDING_MODEL)
    embedding_model_type: Optional[str] = Field(default=None)
    default_embedding_dim: Optional[int] = Field(default=None)
    vicinity_metric: Optional[str] = Field(default=None)

    chunker_type: str = Field(default=common_utils.DEFAULT_CHUNKER_TYPE)
    num_processing_workers: int = Field(default=common_utils.DEFAULT_CHUNK_PROCESSING_WORKERS)
    
    # Semchunk specific
    semchunk_max_tokens_chunk: int = Field(default=common_utils.CHUNKER_DEFAULTS["semchunk"]["max_tokens_chunk"])
    semchunk_overlap_percent: int = Field(default=common_utils.CHUNKER_DEFAULTS["semchunk"]["overlap_percent"])
    
    # Chonkie common (SDPM, Semantic)
    chonkie_embedding_model: str = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["embedding_model"])
    chonkie_target_chunk_size: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["target_chunk_size"])
    chonkie_similarity_threshold: Union[float, int, str] = Field(default_factory=lambda: common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["threshold"]) # Use factory for mutable default
    chonkie_min_sentences: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_sentences"])
    
    # Chonkie SDPM specific
    chonkie_sdpm_mode: str = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["mode"])
    chonkie_sdpm_similarity_window: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["similarity_window"])
    chonkie_sdpm_min_chunk_size: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_chunk_size"])
    chonkie_skip_window: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["skip_window"])
    
    # Chonkie Semantic specific (can reuse some general chonkie params or have its own)
    chonkie_semantic_mode: str = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_semantic"]["mode"])
    chonkie_semantic_similarity_window: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_semantic"]["similarity_window"])
    chonkie_semantic_min_chunk_size: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_semantic"]["min_chunk_size"])

    # Chonkie Neural specific
    chonkie_neural_model: str = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["segmentation_model"])
    chonkie_neural_stride: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["stride"])
    chonkie_neural_min_chars: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["min_chars_per_chunk"])

    # Chonkie Basic (Recursive/Sentence/Token)
    chonkie_basic_tokenizer: str = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_recursive"]["tokenizer_or_token_counter"])
    chonkie_basic_chunk_size: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_recursive"]["chunk_size"])
    chonkie_basic_min_chars: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_recursive"]["min_characters_per_chunk"])
    chonkie_sentence_overlap: int = Field(default=common_utils.CHUNKER_DEFAULTS["chonkie_sentence"]["chunk_overlap"])


    enable_classification: bool = Field(default=False)
    classifier_model_name: Optional[str] = Field(default=common_utils.DEFAULT_CLASSIFIER_MODEL)
    candidate_labels: Optional[List[str]] = Field(default_factory=lambda: list(common_utils.DEFAULT_CLASSIFICATION_LABELS))
    classification_batch_size: int = Field(default=common_utils.DEFAULT_CLASSIFICATION_BATCH_SIZE)
    
    prepare_viz_data: bool = Field(default=False)
    viz_output_file: str = Field(default=common_utils.VISUALIZATION_DATA_FILENAME)
    umap_neighbors: int = Field(default=common_utils.DEFAULT_UMAP_NEIGHBORS)
    umap_min_dist: float = Field(default=common_utils.DEFAULT_UMAP_MIN_DIST)
    umap_metric: str = Field(default=common_utils.DEFAULT_UMAP_METRIC)

# Other Pydantic models (QueryRequest, RerankRequest, etc.) are unchanged.
class QueryRequest(BaseModel): query: str; top_k: int = Field(common_utils.DEFAULT_RAG_INITIAL_K, ge=1, le=200); threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
class RerankRequest(BaseModel): query: str; retrieved_chunks: List[Dict[str, Any]]; top_n_rerank: int = Field(common_utils.DEFAULT_RERANKER_TOP_N, ge=1, le=50)
class StatusResponse(BaseModel): status: str; message: Optional[str]=None; store_loaded: bool; store_items_count: Optional[int]=None; store_metadata: Optional[Dict[str,Any]]=None; reranker_model_name: Optional[str]=None; query_embedding_model_name: Optional[str]=None
class SearchResultItem(BaseModel): id: str; text: Optional[str]=None; score: float; metadata: Optional[Dict[str,Any]]=None
class SearchResponse(BaseModel): query: str; results: List[SearchResultItem]
class VisualizationDataPoint(BaseModel): id: str; x: float; y: float; source_file: Optional[str]=None; display_source: str; classification: Optional[str]=None; snippet: str
class VisualizationDataResponse(BaseModel): data_points: List[VisualizationDataPoint]; message: Optional[str]=None


def run_processing_pipeline_background_task(config_dict: dict): # Unchanged, uses orchestrator
    global app_state; print(f"Background Task: START Data Processing via Orchestrator...")
    success = pipeline_orchestrator.run_full_processing_pipeline(config_dict)
    if success:
        print("Background Task: Orchestrated Pipeline FINISHED. Reloading store...")
        load_config = PipelineRunRequest(vector_data_dir=config_dict.get("vector_data_dir", app_state.pipeline_default_vector_data_dir), gpu_device_id=config_dict.get("gpu_device_id", common_utils.DEFAULT_GPU_DEVICE_ID_PROCESSING))
        load_vector_store_and_data(load_config); print("Background Task: Store reloaded.")
    else: print("Background Task: Orchestrated Pipeline FAILED.")

@app.on_event("startup")
async def startup_event(): # Unchanged
    print("API Startup: Initializing..."); global app_state
    data_dir = os.getenv("PIPELINE_VECTOR_DATA_DIR", app_state.pipeline_default_vector_data_dir)
    gpu_id = int(os.getenv("PIPELINE_GPU_DEVICE_ID", str(common_utils.DEFAULT_GPU_DEVICE_ID_PROCESSING)))
    load_vector_store_and_data(PipelineRunRequest(vector_data_dir=data_dir, gpu_device_id=gpu_id))
    reranker_name = os.getenv("PIPELINE_RERANKER_MODEL", app_state.pipeline_reranker_model_name)
    app_state.reranker_instance = reranker_module.Reranker(reranker_name, device=app_state.embedding_device)
    print(f"API Startup: Initialized. Store loaded: {app_state.vector_store_manager is not None}")

def load_vector_store_and_data(config: PipelineRunRequest): # Unchanged
    global app_state; print(f"API: Loading store from: {config.vector_data_dir}")
    app_state.vector_store_manager = VectorStoreManager(config.vector_data_dir, common_utils.VECTOR_STORE_SUBDIR_NAME)
    if not app_state.vector_store_manager.load_store():
        app_state.vector_store_manager = None; app_state.supporting_data = {"text_chunks":None,"chunk_ids":None,"detailed_metadata":None}
        print("API: Failed to load vector store."); return False
    store_meta = app_state.vector_store_manager.get_store_metadata()
    if store_meta:
        app_state.query_embedding_model_name=store_meta.get("embedding_model_name", common_utils.DEFAULT_QUERY_EMBEDDING_MODEL)
        app_state.query_embedding_model_type=store_meta.get("embedding_model_type"); app_state.query_embedding_model_class=store_meta.get("embedding_model_class")
        app_state.query_embedding_model_dim=store_meta.get("embedding_vector_dimension")
    else:
        app_state.query_embedding_model_type, app_state.query_embedding_model_dim, _, app_state.query_embedding_model_class = \
            embedding_module.infer_embedding_model_params(app_state.query_embedding_model_name)[:4]
    if config.gpu_device_id >= 0:
        if torch.cuda.is_available(): app_state.embedding_device = f"cuda:{config.gpu_device_id}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): app_state.embedding_device = "mps"
        else: app_state.embedding_device = "cpu"; print(f"API: GPU {config.gpu_device_id} for query embed unavailable.")
    else: app_state.embedding_device = "cpu"
    print(f"API: Query model '{app_state.query_embedding_model_name}' on '{app_state.embedding_device}'.")
    tc_p = os.path.join(config.vector_data_dir, common_utils.TEXT_CHUNKS_FILENAME); ci_p = os.path.join(config.vector_data_dir, common_utils.CHUNK_IDS_FILENAME); dm_p = os.path.join(config.vector_data_dir, common_utils.DETAILED_CHUNK_METADATA_FILENAME)
    if all(os.path.exists(p) for p in [tc_p, ci_p, dm_p]):
        try:
            with open(tc_p,'r',encoding='utf-8') as f: app_state.supporting_data["text_chunks"]=json.load(f)
            with open(ci_p,'r',encoding='utf-8') as f: app_state.supporting_data["chunk_ids"]=json.load(f)
            with open(dm_p,'r',encoding='utf-8') as f: app_state.supporting_data["detailed_metadata"]=json.load(f)
            print(f"API: Supporting data loaded ({len(app_state.supporting_data.get('text_chunks',[]))} chunks).")
        except Exception as e: print(f"API Error loading sup. data: {e}"); app_state.supporting_data = {"text_chunks":None,"chunk_ids":None,"detailed_metadata":None}
    else: print("API Warning: Sup. data files missing."); app_state.supporting_data = {"text_chunks":None,"chunk_ids":None,"detailed_metadata":None}
    return True

@app.post("/pipeline/run", status_code=202) # Unchanged
async def run_pipeline_endpoint(request: PipelineRunRequest, background_tasks: BackgroundTasks):
    print(f"API: Received request to run pipeline for docs: {request.docs_folder}")
    config_dict = request.model_dump(); background_tasks.add_task(run_processing_pipeline_background_task, config_dict)
    return {"message": "Data processing (via orchestrator) started in background. Check server logs."}

# /status, /search, /rerank, /visualization_data, /get_chunk_details endpoints remain unchanged from the previous full output
# as they interact with app_state which is loaded by load_vector_store_and_data or updated by the background task.
@app.post("/pipeline/load_data", response_model=StatusResponse)
async def endpoint_load_data(request: PipelineRunRequest = Body(...)):
    if load_vector_store_and_data(request): return await get_service_status()
    else: raise HTTPException(status_code=500, detail="Failed to load vector store or data.")
@app.get("/status", response_model=StatusResponse)
async def get_service_status():
    global app_state; store_loaded = app_state.vector_store_manager is not None
    items_count = app_state.vector_store_manager.get_total_items() if store_loaded and app_state.vector_store_manager else None
    store_meta = app_state.vector_store_manager.get_store_metadata() if store_loaded and app_state.vector_store_manager else None
    reranker_name = app_state.reranker_instance.model.tokenizer.name_or_path if app_state.reranker_instance and app_state.reranker_instance.model else "Not loaded"
    return StatusResponse(status="Service running.", message="Store loaded." if store_loaded else "Store not loaded.", store_loaded=store_loaded, store_items_count=items_count, store_metadata=store_meta, reranker_model_name=reranker_name, query_embedding_model_name=app_state.query_embedding_model_name)
@app.post("/search", response_model=SearchResponse)
async def search_chunks(request: QueryRequest):
    global app_state
    if not app_state.vector_store_manager: raise HTTPException(status_code=503, detail="Vector store not loaded.")
    if not app_state.query_embedding_model_name or not app_state.query_embedding_model_type or app_state.query_embedding_model_dim is None: raise HTTPException(status_code=503, detail="Query embedding model params not configured.")
    query_vec_np, _ = embedding_module.generate_embeddings( [request.query], app_state.query_embedding_model_name, app_state.query_embedding_model_type, app_state.query_embedding_model_dim, app_state.query_embedding_model_class, device=app_state.embedding_device)
    if query_vec_np.shape[0] == 0: raise HTTPException(status_code=500, detail="Failed to generate query embedding.")
    results = app_state.vector_store_manager.query_store(query_vec_np[0], k=request.top_k, threshold=request.threshold)
    search_results_items: List[SearchResultItem] = []
    if app_state.supporting_data["text_chunks"] and app_state.supporting_data["chunk_ids"] and app_state.supporting_data["detailed_metadata"]:
        id_to_idx = {id_val: idx for idx, id_val in enumerate(app_state.supporting_data["chunk_ids"])}
        for item_id, score in results:
            idx = id_to_idx.get(item_id)
            if idx is not None and idx < len(app_state.supporting_data["text_chunks"]) and idx < len(app_state.supporting_data["detailed_metadata"]):
                search_results_items.append(SearchResultItem(id=item_id, text=app_state.supporting_data["text_chunks"][idx], score=score, metadata=app_state.supporting_data["detailed_metadata"][idx]))
            else: search_results_items.append(SearchResultItem(id=item_id, score=score, metadata={"error": "Chunk data missing"}))
    else: search_results_items = [SearchResultItem(id=item_id, score=score) for item_id, score in results]
    return SearchResponse(query=request.query, results=search_results_items)
@app.post("/rerank", response_model=List[SearchResultItem])
async def rerank_provided_chunks(request: RerankRequest):
    global app_state
    if not app_state.reranker_instance or not app_state.reranker_instance.model: raise HTTPException(status_code=503, detail="Reranker model not loaded.")
    chunks_to_rerank: List[Tuple[str, str]] = []; id_to_orig_meta: Dict[str, Any] = {}
    for ch_data in request.retrieved_chunks:
        if ch_data.get("id") and ch_data.get("text"): chunks_to_rerank.append((ch_data["id"], ch_data["text"])); id_to_orig_meta[ch_data["id"]] = {"score": ch_data.get("score"), "metadata": ch_data.get("metadata")}
    if not chunks_to_rerank: return []
    reranked_id_scores: List[Tuple[str, float]] = app_state.reranker_instance.rerank(request.query, chunks_to_rerank, request.top_n_rerank)
    response_items: List[SearchResultItem] = []
    for item_id, new_score in reranked_id_scores:
        orig_text = next((text for cid, text in chunks_to_rerank if cid == item_id), None)
        orig_meta_info = id_to_orig_meta.get(item_id, {})
        response_items.append(SearchResultItem(id=item_id, text=orig_text, score=new_score, metadata=orig_meta_info.get("metadata")))
    return response_items
@app.get("/visualization_data", response_model=VisualizationDataResponse)
async def get_visualization_data(vector_data_dir: str = Query(app_state.pipeline_default_vector_data_dir)):
    viz_file_path = os.path.join(vector_data_dir, common_utils.VISUALIZATION_DATA_FILENAME)
    if not os.path.exists(viz_file_path): raise HTTPException(status_code=404, detail="Visualization data file not found.")
    try:
        with open(viz_file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        return VisualizationDataResponse(data_points=data)
    except Exception as e: raise HTTPException(status_code=500, detail=f"Error reading viz data: {e}")
@app.get("/get_chunk_details/{chunk_id}", response_model=Dict[str, Any])
async def get_chunk_details(chunk_id: str):
    global app_state
    if not all(app_state.supporting_data.get(k) for k in ["chunk_ids", "text_chunks", "detailed_metadata"]): raise HTTPException(status_code=503, detail="Supporting chunk data not loaded.")
    try:
        idx = app_state.supporting_data["chunk_ids"].index(chunk_id)
        return {"id": chunk_id, "text": app_state.supporting_data["text_chunks"][idx], "metadata": app_state.supporting_data["detailed_metadata"][idx]}
    except ValueError: raise HTTPException(status_code=404, detail=f"Chunk ID '{chunk_id}' not found.")
    except IndexError: raise HTTPException(status_code=500, detail=f"Data inconsistency for chunk ID '{chunk_id}'.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")