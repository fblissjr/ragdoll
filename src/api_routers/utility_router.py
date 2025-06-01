# src/api_routers/utility_router.py
# Router for utility API endpoints like service status and visualization data.

import os
import json
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any

# Import shared app_state and Pydantic models
from ..main_api import app_state # Access global app state
from ..api_models import StatusResponse, VisualizationDataResponse, VisualizationDataPoint
from .. import ragdoll_config # For default filenames

router = APIRouter()

# Internal helper to avoid circular dependency if other routers need status
async def get_service_status_internal() -> StatusResponse:
    """Helper to construct status response, callable internally."""
    store_is_loaded = app_state.vector_store_manager is not None
    items_count = None
    store_meta = None
    if store_is_loaded and app_state.vector_store_manager:
        items_count = app_state.vector_store_manager.get_total_items()
        store_meta = app_state.vector_store_manager.get_store_metadata()
        
    reranker_name = "Not Loaded"
    if app_state.reranker_instance and app_state.reranker_instance.model:
        reranker_name = getattr(getattr(app_state.reranker_instance.model, 'tokenizer', None), 'name_or_path', "Unknown Reranker")

    return StatusResponse(
        status="Service is running.", 
        message="Vector store is loaded." if store_is_loaded else "Vector store not loaded.",
        store_loaded=store_is_loaded, 
        store_items_count=items_count, 
        store_metadata=store_meta, 
        reranker_model_name=reranker_name, 
        query_embedding_model_name=app_state.query_embedding_model_name,
        query_embedding_device=app_state.embedding_device
    )


@router.get("/status", response_model=StatusResponse, summary="Get API Service Status")
async def endpoint_get_service_status():
    """
    Returns the current status of the data service, including vector store information
    and configured model details.
    """
    return await get_service_status_internal()


@router.get("/visualization_data", response_model=VisualizationDataResponse, summary="Get UMAP Visualization Data")
async def endpoint_get_visualization_data(
    vector_data_dir: str = Query(
        default=app_state.pipeline_default_vector_data_dir, # Uses default from app_state
        description="Directory where the visualization JSON file is stored."
    )
):
    """
    Retrieves pre-computed UMAP 2D projection data.
    The data is expected in a JSON file (default: visualization_plot_data.json).
    """
    viz_file_path = os.path.join(vector_data_dir, ragdoll_config.VISUALIZATION_DATA_FILENAME)
    if not os.path.exists(viz_file_path): 
        raise HTTPException(status_code=404, detail=f"Visualization data file not found at: {viz_file_path}")
    
    try:
        with open(viz_file_path, 'r', encoding='utf-8') as f: 
            data = json.load(f)
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise HTTPException(status_code=500, detail="Visualization data file content is not a valid list of data points.")
        # Pydantic will validate each item against VisualizationDataPoint
        return VisualizationDataResponse(data_points=data) 
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error decoding JSON from visualization data file.")
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"Error reading visualization data: {str(e)}")