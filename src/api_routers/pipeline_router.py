# src/api_routers/pipeline_router.py
# Router for API endpoints related to managing and triggering the data processing pipeline.

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException
from typing import Dict

# Import shared app_state, helper functions, and Pydantic models
# Assuming main_api.py is in the parent directory (src)
from ..main_api import app_state, run_processing_pipeline_background_task, load_vector_store_and_data 
from ..api_models import PipelineRunRequest, StatusResponse, MessageResponse

router = APIRouter()

@router.post("/run", status_code=202, response_model=MessageResponse, summary="Run Data Processing Pipeline")
async def endpoint_run_pipeline(
    request: PipelineRunRequest, 
    background_tasks: BackgroundTasks
):
    """
    Asynchronously triggers the full data processing pipeline (document ingestion,
    chunking, embedding, vector store creation).
    The actual processing happens in the background.
    """
    print(f"Pipeline Router (/run): Received request for docs folder: {request.docs_folder}")
    if not os.path.exists(request.docs_folder) or not os.listdir(request.docs_folder):
        raise HTTPException(status_code=400, detail=f"Documents folder '{request.docs_folder}' is empty or does not exist.")

    pipeline_config_dict = request.model_dump()
    background_tasks.add_task(run_processing_pipeline_background_task, pipeline_config_dict)
    return MessageResponse(message="Data processing pipeline started in background. Monitor server logs for progress.")

@router.post("/load_data", response_model=StatusResponse, summary="Load Processed Data from Directory")
async def endpoint_load_data(
    request: PipelineRunRequest = Body(
        ..., 
        description="Configuration specifying the data directory to load from. Only vector_data_dir and gpu_device_id are primarily used from this request for loading."
    )
):
    """
    Manually triggers a reload of the vector store and supporting data from a specified directory.
    This is useful if data was processed offline or needs to be refreshed in the service.
    Note: Only vector_data_dir and gpu_device_id from the request are used for loading.
    Other pipeline parameters are ignored for this specific endpoint.
    """
    print(f"Pipeline Router (/load_data): Received request to load data from: {request.vector_data_dir}")
    # The load_vector_store_and_data function expects a PipelineRunRequest object
    # It primarily uses vector_data_dir and gpu_device_id from it for this loading operation.
    if load_vector_store_and_data(request):
        # To return a full StatusResponse, we need access to the get_service_status logic.
        # For simplicity now, we'll construct a basic success message.
        # A more robust solution might involve a shared status utility or dependency injection.
        loaded_status = await utility_router.get_service_status_internal() # Call internal status getter
        return loaded_status
    else:
        raise HTTPException(status_code=500, detail=f"Failed to load vector store or supporting data from '{request.vector_data_dir}'.")