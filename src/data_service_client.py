# src/data_service_client.py
# Client for interacting with the RAGdoll API Service.

import requests 
import json     
from typing import List, Dict, Optional, Any, Tuple, Union
from pydantic import BaseModel # For structuring RerankRequestItemClient

# Project-specific configuration for the default service URL
from . import ragdoll_config 

# Client-side Pydantic model for items in rerank request to ensure structure
# This helps ensure the client sends data in the format the server expects for reranking.
class RerankRequestItemClient(BaseModel):
    id: str
    text: str
    score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

class DataServiceClient:
    def __init__(self, base_url: Optional[str] = None):
        """
        Initializes the DataServiceClient.
        Args:
            base_url: Base URL of the RAGdoll API Service. Uses config default if None.
        """
        self.base_url = base_url or ragdoll_config.DEFAULT_DATA_SERVICE_URL
        print(f"Data Service Client: Initialized. API URL: {self.base_url}")

    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict[str, Any]] = None, 
        json_data: Optional[Dict[str, Any]] = None,
        timeout: int = 90 # Increased default timeout for potentially longer operations
    ) -> Optional[Any]: # Can return Dict or List[Dict] based on endpoint
        """Helper to make HTTP requests to the API service."""
        full_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            response = None
            if method.upper() == "GET":
                response = requests.get(full_url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(full_url, params=params, json=json_data, timeout=timeout)
            else:
                print(f"DataServiceClient Error: Unsupported HTTP method '{method}' for {full_url}")
                return None
            
            response.raise_for_status()
            # Check if response content is empty before trying to parse JSON
            if response.content:
                return response.json()
            else: # Handle empty response (e.g. 204 No Content, or successful POST that returns nothing)
                return {"status": "success", "message": "Request successful with no content returned."} if response.ok else None


        except requests.exceptions.RequestException as e:
            print(f"DataServiceClient Error: Request to {full_url} ({method.upper()}) failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try: 
                    print(f"  Response Status: {e.response.status_code}, Content: {e.response.text[:500]}...")
                except: pass
            return None
        except json.JSONDecodeError as e:
            print(f"DataServiceClient Error: Failed to decode JSON from {full_url}: {e}")
            if 'response' in locals() and response is not None:
                 print(f"  Raw Response Text: {response.text[:500]}...")
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        """Fetches the status of the data service."""
        return self._make_request("GET", "/service/status") # Updated endpoint prefix

    def search(self, query: str, top_k: int, threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Sends a search request."""
        payload = {"query": query, "top_k": top_k}
        if threshold is not None: payload["threshold"] = threshold
        return self._make_request("POST", "/data/search", json_data=payload) # Updated endpoint

    # Type hint for retrieved_chunks_data uses Dict for flexibility, client ensures structure
    def rerank(self, query: str, retrieved_chunks_data: List[Dict[str, Any]], top_n_rerank: int) -> Optional[List[Dict[str, Any]]]:
        """Sends a rerank request. retrieved_chunks_data should be list of RerankRequestItemClient-like dicts."""
        # Client can do a basic validation or rely on Pydantic model on server for full validation
        valid_chunks_for_api = []
        for chunk in retrieved_chunks_data:
            if isinstance(chunk, dict) and "id" in chunk and "text" in chunk:
                valid_chunks_for_api.append(chunk) # Pass as dict; server Pydantic model will parse
            else:
                print(f"DataServiceClient Warning: Skipping invalid chunk for reranking: {str(chunk)[:100]}")
        
        if not valid_chunks_for_api: return []

        payload = {"query": query, "retrieved_chunks": valid_chunks_for_api, "top_n_rerank": top_n_rerank}
        return self._make_request("POST", "/data/rerank", json_data=payload) # Updated endpoint

    def get_chunk_details(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Fetches details for a specific chunk ID."""
        return self._make_request("GET", f"/data/get_chunk_details/{chunk_id}") # Updated endpoint
        
    def get_visualization_data(self, vector_data_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Fetches UMAP visualization data."""
        params: Dict[str, Any] = {}
        if vector_data_dir: params["vector_data_dir"] = vector_data_dir
        return self._make_request("GET", "/service/visualization_data", params=params) # Updated endpoint

    def trigger_pipeline_run(self, pipeline_config_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Triggers a data processing pipeline run on the server."""
        return self._make_request("POST", "/pipeline/run", json_data=pipeline_config_payload, timeout=30) 
    
    def trigger_load_data(self, load_config_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Triggers a data load operation on the server."""
        # Expects relevant parts of PipelineRunRequest, e.g., {"vector_data_dir": "...", "gpu_device_id": 0}
        return self._make_request("POST", "/pipeline/load_data", json_data=load_config_payload)