# llm_interface.py
import requests
import json
from typing import List, Dict, Optional, Any

class LLMInterface:
    def __init__(
        self, 
        api_url: str, 
        model_name: str = "mlx-community", 
        default_max_tokens: int = 4096,
        default_temperature: float = 1,
        default_streaming: bool = True # Server might not support true streaming via simple requests
    ):
        self.api_url = api_url
        self.model_name = model_name # This is for the OpenAI API spec, mlx-lm server ignores it
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_streaming = default_streaming
        print(f"LLM Interface: Initialized for API URL: {self.api_url}, Model (placeholder): {self.model_name}")

    def generate_chat_completion(
        self, 
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: Optional[bool] = None # Currently, direct requests won't handle true streaming easily
    ) -> Optional[str]:
        
        max_tokens_to_use = max_tokens if max_tokens is not None else self.default_max_tokens
        temp_to_use = temperature if temperature is not None else self.default_temperature
        stream_to_use = stream if stream is not None else self.default_streaming

        if stream_to_use:
            print("LLM Interface: Note - True streaming via simple requests is not fully implemented. Will fetch full response.")
        
        payload = {
            "model": self.model_name, # Ignored by mlx-lm server, but part of OpenAI spec
            "messages": messages,
            "max_tokens": max_tokens_to_use,
            "temperature": temp_to_use,
            "stream": False # Force False for simple requests; true streaming needs different client logic
        }
        headers = {"Content-Type": "application/json"}

        print(f"LLM Interface: Sending request to {self.api_url} with max_tokens={max_tokens_to_use}, temp={temp_to_use}")
        # print(f"  Payload Messages Preview: {json.dumps(messages[:1], indent=2)} ... (Total: {len(messages)})")

        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120) # 120s timeout
            response.raise_for_status() 
            
            response_data = response.json()
            
            if response_data.get("choices") and len(response_data["choices"]) > 0:
                first_choice = response_data["choices"][0]
                if first_choice.get("message") and first_choice["message"].get("content"):
                    return first_choice["message"]["content"].strip()
                elif first_choice.get("delta") and first_choice["delta"].get("content"): # For some stream-like non-streaming formats
                    return first_choice["delta"]["content"].strip()
            
            print(f"LLM Interface: Unexpected response structure: {json.dumps(response_data, indent=2)}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"LLM Interface: API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try: print(f"  Response Content: {e.response.text}")
                except: pass
            return None
        except json.JSONDecodeError as e:
            print(f"LLM Interface: Failed to decode JSON response: {e}")
            print(f"  Raw Response Text: {response.text if 'response' in locals() else 'No response object'}")
            return None
        except Exception as e:
            print(f"LLM Interface: An unexpected error occurred: {e}")
            return None