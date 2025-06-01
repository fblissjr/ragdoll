# src/main_api.py
# Main FastAPI application for the RAGdoll service.
# Initializes the app, manages global state, includes API routers, and handles startup events.

import os
import json
import torch # For checking CUDA and MPS availability
from fastapi import FastAPI, BackgroundTasks, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List

# Project-specific imports
from . import ragdoll_config
from . import ragdoll_utils # Placeholder, may not be directly used here but good to have structure
from . import embedding_module
from . import vector_store_manager
from . import reranker_module
from . import pipeline_orchestrator

# Import Pydantic models from the new api_models module
from .api_models import (
    PipelineRunRequest, QueryRequest, RerankRequest, StatusResponse, 
    SearchResultItem, SearchResponse, ChunkDetailResponse, 
    VisualizationDataResponse, MessageResponse
)

# Import API routers (will be created in the next step)
from .api_routers import pipeline_router, retrieval_router, utility_router

# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAGdoll API Service",
    version="0.5.0", # Reflects refactoring
    description="Provides API endpoints for RAGdoll: document processing, vector search, RAG, and data exploration utilities."
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production: specify frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Application State ---
# Global state shared across API requests and modules if necessary.
class AppState:
    vector_store_manager: Optional[vector_store_manager.VectorStoreManager] = None
    reranker_instance: Optional[reranker_module.Reranker] = None
    supporting_data: Dict[str, Any] = {
        "text_chunks": None, "chunk_ids": None, "detailed_metadata": None
    }
    query_embedding_model_name: str = ragdoll_config.DEFAULT_QUERY_EMBEDDING_MODEL
    query_embedding_model_type: Optional[str] = None
    query_embedding_model_class: Optional[str] = None
    query_embedding_model_dim: Optional[int] = None
    embedding_device: str = "cpu" # For query-time embeddings and reranker
    # This default path should be consistent with how PipelineRunRequest default is handled.
    # Consider making pipeline_default_vector_data_dir a direct config constant.
    pipeline_default_vector_data_dir: str = "vector_store_data" 
    pipeline_reranker_model_name: str = ragdoll_config.DEFAULT_RERANKER_MODEL

app_state = AppState() # Global instance of the application state

# --- Helper Functions (Potentially moved to an api_utils.py later if they grow) ---

def load_vector_store_and_data(config_for_loading: PipelineRunRequest):
    """
    Loads vector store and supporting data into app_state.
    Configures query embedding parameters and device.
    (Content is the same as in the previous `fastapi_data_service.py` version,
     but now uses app_state and imports from ragdoll_config)
    """
    global app_state
    vector_data_dir = config_for_loading.vector_data_dir
    print(f"Main API: Attempting to load vector store and data from: {vector_data_dir}")
    
    app_state.vector_store_manager = vector_store_manager.VectorStoreManager(
        vector_data_dir, ragdoll_config.VECTOR_STORE_SUBDIR_NAME
    )
    if not app_state.vector_store_manager.load_store():
        app_state.vector_store_manager = None
        app_state.supporting_data = {"text_chunks":None,"chunk_ids":None,"detailed_metadata":None}
        print(f"Main API: Failed to load vector store from '{vector_data_dir}'."); return False
    
    store_meta = app_state.vector_store_manager.get_store_metadata()
    if store_meta:
        app_state.query_embedding_model_name = store_meta.get("embedding_model_name", ragdoll_config.DEFAULT_QUERY_EMBEDDING_MODEL)
        app_state.query_embedding_model_type = store_meta.get("embedding_model_type")
        app_state.query_embedding_model_class = store_meta.get("embedding_model_class")
        app_state.query_embedding_model_dim = store_meta.get("embedding_vector_dimension")
    else:
        print(f"Main API: Store metadata incomplete. Inferring query model params for '{app_state.query_embedding_model_name}'.")
        app_state.query_embedding_model_type, app_state.query_embedding_model_dim, _, app_state.query_embedding_model_class = \
            embedding_module.infer_embedding_model_params(app_state.query_embedding_model_name)[:4]

    gpu_id_for_query = config_for_loading.gpu_device_id
    if gpu_id_for_query >= 0:
        if torch.cuda.is_available(): app_state.embedding_device = f"cuda:{gpu_id_for_query}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): app_state.embedding_device = "mps"
        else: app_state.embedding_device = "cpu"; print(f"Main API: GPU {gpu_id_for_query} for query device unavailable, using CPU.")
    else: app_state.embedding_device = "cpu"
    print(f"Main API: Query model '{app_state.query_embedding_model_name}' on device '{app_state.embedding_device}'.")
    
    # Load supporting data
    tc_p = os.path.join(vector_data_dir, ragdoll_config.TEXT_CHUNKS_FILENAME)
    ci_p = os.path.join(vector_data_dir, ragdoll_config.CHUNK_IDS_FILENAME)
    dm_p = os.path.join(vector_data_dir, ragdoll_config.DETAILED_CHUNK_METADATA_FILENAME)
    if all(os.path.exists(p) for p in [tc_p, ci_p, dm_p]):
        try:
            with open(tc_p,'r',encoding='utf-8') as f: app_state.supporting_data["text_chunks"]=json.load(f)
            with open(ci_p,'r',encoding='utf-8') as f: app_state.supporting_data["chunk_ids"]=json.load(f)
            with open(dm_p,'r',encoding='utf-8') as f: app_state.supporting_data["detailed_metadata"]=json.load(f)
            print(f"Main API: Supporting data loaded ({len(app_state.supporting_data.get('text_chunks',[]))} chunks).")
        except Exception as e: 
            print(f"Main API Error: Loading supporting data from '{vector_data_dir}': {e}"); 
            app_state.supporting_data = {"text_chunks":None,"chunk_ids":None,"detailed_metadata":None}
    else: 
        print(f"Main API Warning: Supporting data files missing in '{vector_data_dir}'."); 
        app_state.supporting_data = {"text_chunks":None,"chunk_ids":None,"detailed_metadata":None}
    return True

def run_processing_pipeline_background_task(pipeline_config_dict: dict):
    """ Background task wrapper for pipeline_orchestrator.run_full_processing_pipeline. """
    global app_state
    print(f"Main API Background Task: Starting Data Processing via Orchestrator for docs: {pipeline_config_dict.get('docs_folder')}")
    success = pipeline_orchestrator.run_full_processing_pipeline(pipeline_config_dict)
    if success:
        print("Main API Background Task: Pipeline FINISHED. Reloading store...")
        load_req = PipelineRunRequest( # Create a Pydantic model instance for type safety
            vector_data_dir=pipeline_config_dict.get("vector_data_dir", app_state.pipeline_default_vector_data_dir), 
            gpu_device_id=pipeline_config_dict.get("gpu_device_id", ragdoll_config.DEFAULT_GPU_DEVICE_ID_PROCESSING)
        )
        load_vector_store_and_data(load_req)
        print("Main API Background Task: Store reloaded post-pipeline.")
    else: 
        print("Main API Background Task: Orchestrated Pipeline FAILED.")

# --- FastAPI Startup Event ---
@app.on_event("startup")
async def startup_event_handler():
    """
    Actions to perform when the FastAPI application starts:
    - Load vector store and supporting data.
    - Initialize reranker model.
    """
    print("Main API: Startup event. Initializing service state...")
    global app_state
    
    data_dir = os.getenv("RAGDOLL_INITIAL_DATA_DIR", app_state.pipeline_default_vector_data_dir)
    gpu_id_for_startup = ragdoll_config.DEFAULT_GPU_DEVICE_ID_PROCESSING # Uses env-aware default
    
    startup_load_req = PipelineRunRequest(vector_data_dir=data_dir, gpu_device_id=gpu_id_for_startup)
    load_vector_store_and_data(startup_load_req)
    
    reranker_model_name = os.getenv("RAGDOLL_RERANKER_MODEL", app_state.pipeline_reranker_model_name)
    app_state.reranker_instance = reranker_module.Reranker(reranker_model_name, device=app_state.embedding_device)
    
    print(f"Main API: Startup complete. Store loaded: {app_state.vector_store_manager is not None}. Reranker: '{reranker_model_name}' on '{app_state.embedding_device}'.")

# --- Include API Routers ---
# Endpoints are now defined in separate router files and included here.
app.include_router(pipeline_router.router, prefix="/pipeline", tags=["Pipeline Management"])
app.include_router(retrieval_router.router, prefix="/data", tags=["Data Retrieval & Reranking"]) # Changed prefix for clarity
app.include_router(utility_router.router, prefix="/service", tags=["Service Utilities"])       # Changed prefix


# --- Main Execution Block (for Uvicorn development server) ---
if __name__ == "__main__":
    import uvicorn
    server_host = os.getenv("RAGDOLL_API_HOST", "0.0.0.0")
    server_port = int(os.getenv("RAGDOLL_API_PORT", "8001"))
    log_level = ragdoll_config.DEFAULT_LOG_LEVEL.lower()

    print(f"Main API: Starting Uvicorn server. Host: {server_host}, Port: {server_port}, LogLevel: {log_level}")
    uvicorn.run(
        "main_api:app", # Points to this file (main_api.py) and the app instance
        host=server_host, 
        port=server_port, 
        log_level=log_level,
        reload=True # Enable auto-reload for development. Remove for production.
    )