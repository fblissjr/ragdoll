# src/llm_interface.py
# Provides a client interface for OpenAI-compatible LLM APIs.

import requests 
import json     
from typing import List, Dict, Optional, Any

# No direct imports from ragdoll_utils or ragdoll_config typically needed here if API URL etc. are init params.
# Defaults for temperature, max_tokens are handled as init params.

class LLMInterface:
    def __init__(
        self, 
        api_url: str, 
        model_name: str = "default-llm-model/placeholder", 
        default_max_tokens: int = 2048, 
        default_temperature: float = 0.5,
        default_timeout_seconds: int = 180 # Increased default timeout
    ):
        """
        Initializes the LLMInterface.
        """
        self.api_url = api_url
        self.model_name = model_name 
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_timeout_seconds = default_timeout_seconds
        print(f"LLM Interface: Initialized for API URL: {self.api_url}, Model (placeholder for API): {self.model_name}")

    def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: Optional[bool] = False # Placeholder for stream, currently not implemented for requests.post
    ) -> Optional[str]:
        """
        Sends a chat completion request to the LLM API.
        Note: 'stream=True' is not fully supported by this simple requests-based client.
        """
        max_tokens_to_use = max_tokens if max_tokens is not None else self.default_max_tokens
        temp_to_use = temperature if temperature is not None else self.default_temperature
        
        if stream:
            print("LLM Interface Note: Streaming requested, but this client uses simple POST and will fetch the full response (stream=False in payload).")
        
        payload = {
            "model": self.model_name, 
            "messages": messages,
            "max_tokens": max_tokens_to_use,
            "temperature": temp_to_use,
            "stream": False # Forcing False for this implementation
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}

        print(f"LLM Interface: Sending request to {self.api_url} (max_tokens={max_tokens_to_use}, temp={temp_to_use}) for {len(messages)} messages.")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.default_timeout_seconds)
            response.raise_for_status() 
            response_data = response.json()
            
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                first_choice = response_data["choices"][0]
                if first_choice.get("message") and first_choice["message"].get("content"):
                    return first_choice["message"]["content"].strip()
                elif first_choice.get("delta") and first_choice["delta"].get("content"): # For some non-streaming "stream-like" formats
                    return first_choice["delta"]["content"].strip()
            
            print(f"LLM Interface Warning: Unexpected response structure: {json.dumps(response_data, indent=2)}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"LLM Interface Error: API request to {self.api_url} failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try: 
                    print(f"  Response Status: {e.response.status_code}, Content: {e.response.text[:500]}...")
                except: pass
            return None
        except json.JSONDecodeError as e:
            print(f"LLM Interface Error: Failed to decode JSON response from {self.api_url}: {e}")
            raw_response_text = response.text if 'response' in locals() and response is not None else "N/A"
            print(f"  Raw Response Text (first 500 chars): {raw_response_text[:500]}...")
            return None
        except Exception as e:
            print(f"LLM Interface Error: Unexpected error during LLM communication: {e}")
            return None