# data_service_client.py
import requests
import json
from typing import List, Dict, Optional, Any, Tuple

class DataServiceClient:
    def __init__(self, base_url: str = "http://localhost:8001"): # Default from common_utils
        self.base_url = base_url
        print(f"Data Service Client: Initialized for base URL: {self.base_url}")

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None, data: Optional[Dict] = None, timeout: int = 60) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, json=data, timeout=timeout)
            else:
                print(f"DataServiceClient Error: Unsupported HTTP method '{method}'")
                return None
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"DataServiceClient Error: Request to {url} failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try: print(f"  Response Status: {e.response.status_code}, Content: {e.response.text[:500]}...")
                except: pass
            return None
        except json.JSONDecodeError as e:
            print(f"DataServiceClient Error: Failed to decode JSON response from {url}: {e}")
            if 'response' in locals() and response is not None:
                 print(f"  Raw Response: {response.text[:500]}...")
            return None

    def get_status(self) -> Optional[Dict[str, Any]]:
        return self._make_request("GET", "/status")

    def search(self, query: str, top_k: int, threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
        payload = {"query": query, "top_k": top_k}
        if threshold is not None:
            payload["threshold"] = threshold
        return self._make_request("POST", "/search", data=payload)

    def rerank(self, query: str, retrieved_chunks: List[Dict[str, Any]], top_n_rerank: int) -> Optional[List[Dict[str, Any]]]:
        payload = {"query": query, "retrieved_chunks": retrieved_chunks, "top_n_rerank": top_n_rerank}
        return self._make_request("POST", "/rerank", data=payload)

    def get_chunk_details(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        return self._make_request("GET", f"/get_chunk_details/{chunk_id}")
        
    def get_visualization_data(self, vector_data_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
        params = {}
        if vector_data_dir:
            params["vector_data_dir"] = vector_data_dir
        return self._make_request("GET", "/visualization_data", params=params)

    def trigger_pipeline_run(self, pipeline_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # pipeline_config should match the Pydantic model PipelineRunRequest on the server
        return self._make_request("POST", "/pipeline/run", data=pipeline_config, timeout=10) # Short timeout for starting task