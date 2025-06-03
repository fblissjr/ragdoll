# src/vector_store_manager.py
import os
import shutil
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np
# from datetime import datetime # Not used directly in this snippet
from vicinity import Vicinity, Backend, Metric # Ensure Vicinity is imported

from . import ragdoll_config # For default store name if not provided

class VectorStoreManager:
    def __init__(self, 
                 vector_data_dir: str, 
                 vicinity_store_name: Optional[str] = None): # Add parameter with default
        self.vector_data_dir = vector_data_dir
        # Use provided name or fall back to default from config
        self.vicinity_store_name = vicinity_store_name or ragdoll_config.VECTOR_STORE_SUBDIR_NAME
        self.vicinity_store_path = os.path.join(self.vector_data_dir, self.vicinity_store_name)
        self.vicinity_instance: Optional[Vicinity] = None # Correct type hint for Vicinity
        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        # Ensures the main vector_data_dir exists.
        # The vicinity_store_path (subdirectory) is handled by Vicinity's save/load or create_and_save_store.
        os.makedirs(self.vector_data_dir, exist_ok=True)

    def create_and_save_store(
        self, 
        vectors: np.ndarray, 
        items: List[str], # Typically chunk_ids
        metric: Metric, 
        processing_metadata_dict: Dict[str, Any], 
        overwrite: bool = False
    ) -> bool:
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[0] == 0:
            print("VectorStoreManager Error: Invalid or empty vectors provided for store creation.")
            return False
        if not items or len(vectors) != len(items):
            print("VectorStoreManager Error: Items list is empty or does not match vector count.")
            return False
        
        if os.path.exists(self.vicinity_store_path):
            if overwrite:
                print(f"VectorStoreManager: Overwriting existing Vicinity store at '{self.vicinity_store_path}'...")
                if os.path.isdir(self.vicinity_store_path): # Vicinity stores are directories
                    shutil.rmtree(self.vicinity_store_path)
                else: # Should not happen if Vicinity always creates dirs
                    os.remove(self.vicinity_store_path) 
            else:
                print(f"VectorStoreManager Error: Store already exists at '{self.vicinity_store_path}' and overwrite is False.")
                return False
        
        try:
            print(f"VectorStoreManager: Initializing new Vicinity store (Items: {len(items)}, Metric: {metric})...")
            # Vicinity.from_vectors_and_items is a good way to create a new store.
            # Ensure backend_type is what you intend (BASIC is default and simple).
            self.vicinity_instance = Vicinity.from_vectors_and_items(
                vectors=vectors,
                items=items,
                backend_type=Backend.BASIC, 
                metric=metric
            )
            # Attach the processing metadata to the Vicinity instance itself.
            # Vicinity stores this in its arguments.json.
            self.vicinity_instance.metadata = processing_metadata_dict 
            
            print(f"VectorStoreManager: Saving Vicinity store to '{self.vicinity_store_path}'...")
            self.vicinity_instance.save(self.vicinity_store_path) # Vicinity handles creating the directory path
            print("VectorStoreManager: Vicinity store saved successfully.")
            return True
        except Exception as e_create_store:
            print(f"VectorStoreManager Error: Failed to create/save Vicinity store: {e_create_store}")
            self.vicinity_instance = None # Ensure instance is None if creation fails
            return False

    def load_store(self) -> bool:
        if not os.path.exists(self.vicinity_store_path):
            print(f"VectorStoreManager Info: Store not found at '{self.vicinity_store_path}'. Cannot load.")
            return False
        try:
            print(f"VectorStoreManager: Loading Vicinity store from '{self.vicinity_store_path}'...")
            self.vicinity_instance = Vicinity.load(self.vicinity_store_path)
            print("VectorStoreManager: Vicinity store loaded successfully.")
            if self.vicinity_instance and self.vicinity_instance.metadata:
                print("  Store Metadata Preview (from arguments.json):")
                for key, value in list(self.vicinity_instance.metadata.items())[:5]: # Preview first 5 items
                    if isinstance(value, (dict, list)) and len(value) > 3: # If complex or long
                        print(f"    {key}: (type: {type(value).__name__}, preview: {str(value)[:100]}...)")
                    else:
                        print(f"    {key}: {value}")
                if len(self.vicinity_instance.metadata) > 5: print("    ...")
            return True
        except Exception as e_load_store:
            print(f"VectorStoreManager Error: Failed to load Vicinity store from '{self.vicinity_store_path}': {e_load_store}")
            self.vicinity_instance = None
            return False

    def query_store(self, query_vector: np.ndarray, k: int = 5, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        if self.vicinity_instance is None:
            print("VectorStoreManager Error: Store not loaded. Cannot perform query.")
            return []
        if not isinstance(query_vector, np.ndarray) or query_vector.ndim != 1:
            print("VectorStoreManager Error: Query vector must be a 1D NumPy array.")
            return []
        
        try:
            # Vicinity's query methods expect a flattened 1D array for a single query.
            query_vector_flat = query_vector.flatten()
            
            if threshold is not None:
                # query_threshold returns List[List[Tuple[Item, Score]]]
                results_wrapper = self.vicinity_instance.query_threshold(query_vector_flat, threshold=threshold)
            else:
                # query returns List[List[Tuple[Item, Score]]]
                results_wrapper = self.vicinity_instance.query(query_vector_flat, k=k)
            
            # For a single query vector, results_wrapper should be a list containing one list of results.
            if results_wrapper and isinstance(results_wrapper, list) and len(results_wrapper) == 1 and isinstance(results_wrapper[0], list):
                actual_results_list = results_wrapper[0]
                return [(str(item_id), float(score)) for item_id, score in actual_results_list]
            else: # Handle unexpected result structure gracefully
                print(f"VectorStoreManager Warning: Unexpected result structure from Vicinity query: {type(results_wrapper)}")
                return []

        except Exception as e_query:
            print(f"VectorStoreManager Error: Failed to query store: {e_query}")
            return []

    def get_store_metadata(self) -> Optional[Dict[str, Any]]:
        if self.vicinity_instance and hasattr(self.vicinity_instance, 'metadata'):
            return self.vicinity_instance.metadata
        return None

    def get_total_items(self) -> int:
        if self.vicinity_instance:
            try:
                return len(self.vicinity_instance) # Vicinity stores should support len()
            except TypeError: # Some backend types might not directly support len() on the main object
                if hasattr(self.vicinity_instance, '_items'): # Check for internal _items list as fallback
                     return len(self.vicinity_instance._items)
        return 0