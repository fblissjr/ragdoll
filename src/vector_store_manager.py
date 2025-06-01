# src/vector_store_manager.py
# Manages interactions with the Vicinity vector store, including creation,
# loading, querying, and metadata handling.

import os
import shutil # For directory operations like rmtree
import json
import numpy as np
from typing import Optional, List, Tuple, Dict, Any # For type hinting

# Vicinity library for vector storage and search
from vicinity import Vicinity, Backend, Metric 

# Project-specific configuration for default store name
from . import ragdoll_config # For VECTOR_STORE_SUBDIR_NAME

class VectorStoreManager:
    def __init__(self, vector_data_dir: str, vicinity_store_subdir_name: str = ragdoll_config.VECTOR_STORE_SUBDIR_NAME):
        """
        Initializes the VectorStoreManager.

        Args:
            vector_data_dir (str): The base directory where vector store data is located.
            vicinity_store_subdir_name (str, optional): The name of the subdirectory within
                vector_data_dir that holds the Vicinity store. Defaults to value from ragdoll_config.
        """
        self.vector_data_dir = vector_data_dir
        self.vicinity_store_directory_name = vicinity_store_subdir_name 
        self.vicinity_store_path = os.path.join(self.vector_data_dir, self.vicinity_store_directory_name)
        self.vicinity_instance: Optional[Vicinity] = None 
        self._ensure_data_dir_exists() 

    def _ensure_data_dir_exists(self):
        """Creates the base vector_data_dir if it doesn't exist."""
        os.makedirs(self.vector_data_dir, exist_ok=True)

    def create_and_save_store(
        self, 
        vectors: np.ndarray, 
        items: List[str], 
        metric: Metric,   
        metadata_to_store: Dict[str, Any], 
        overwrite: bool = False 
    ) -> bool:
        """
        Creates a new Vicinity vector store and saves it to disk.
        """
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[0] == 0:
            print("VectorStoreManager Error: Invalid or empty vectors array. Store not created.")
            return False
        if not items or len(vectors) != len(items):
            print("VectorStoreManager Error: Items list mismatch with vector count. Store not created.")
            return False
        
        if os.path.exists(self.vicinity_store_path):
            if overwrite:
                print(f"VectorStoreManager: Overwriting existing store at '{self.vicinity_store_path}'...")
                if os.path.isdir(self.vicinity_store_path):
                    shutil.rmtree(self.vicinity_store_path)
                else: 
                    os.remove(self.vicinity_store_path) 
            else:
                print(f"VectorStoreManager Error: Store exists at '{self.vicinity_store_path}' and overwrite is False.")
                return False
        
        try:
            print(f"VectorStoreManager: Initializing Vicinity store. Items: {len(items)}, Metric: {metric}.")
            self.vicinity_instance = Vicinity.from_vectors_and_items(
                vectors=vectors.astype(np.float32), 
                items=items,
                backend_type=Backend.BASIC, 
                metric=metric
            )
            self.vicinity_instance.metadata = metadata_to_store
            
            print(f"VectorStoreManager: Saving store to '{self.vicinity_store_path}'...")
            self.vicinity_instance.save(self.vicinity_store_path) 
            print("VectorStoreManager: Vicinity store saved successfully.")
            return True
        except Exception as e: 
            print(f"VectorStoreManager Error: Failed to create/save Vicinity store: {e}")
            self.vicinity_instance = None 
            return False

    def load_store(self) -> bool:
        """
        Loads an existing Vicinity vector store from disk.
        """
        if not os.path.exists(self.vicinity_store_path) or not os.path.isdir(self.vicinity_store_path):
            print(f"VectorStoreManager Info: Store not found at '{self.vicinity_store_path}'.")
            self.vicinity_instance = None
            return False
        try:
            print(f"VectorStoreManager: Loading Vicinity store from '{self.vicinity_store_path}'...")
            self.vicinity_instance = Vicinity.load(self.vicinity_store_path)
            num_items = len(self.vicinity_instance) if self.vicinity_instance else 0
            print(f"VectorStoreManager: Store loaded with {num_items} items.")
            
            if self.vicinity_instance and self.vicinity_instance.metadata:
                print("  Loaded Store Metadata (Preview):")
                for key, value in self.vicinity_instance.metadata.items():
                    if isinstance(value, (dict, list)) and len(str(value)) > 150:
                        preview = str(list(value.keys())[:3] if isinstance(value, dict) else value[:3]) + "..."
                        print(f"    {key}: (Preview: {preview})")
                    else:
                        print(f"    {key}: {value}")
            elif self.vicinity_instance:
                 print("  Store loaded but has no additional metadata attached.")
            return True
        except Exception as e: 
            print(f"VectorStoreManager Error: Failed to load Vicinity store '{self.vicinity_store_path}': {e}")
            self.vicinity_instance = None 
            return False

    def query_store(self, query_vector: np.ndarray, k: int = 5, threshold: Optional[float] = None) -> List[Tuple[str, float]]:
        """
        Queries the loaded Vicinity store.
        """
        if self.vicinity_instance is None:
            print("VectorStoreManager Error: Store not loaded. Cannot query.")
            return []
        if not isinstance(query_vector, np.ndarray) or query_vector.ndim != 1:
            print("VectorStoreManager Error: Query vector must be a 1D NumPy array.")
            return []
        
        try:
            query_vector_2d = query_vector.astype(np.float32).reshape(1, -1)
            results_from_vicinity: List[List[Tuple[Any, float]]] 
            if threshold is not None:
                results_from_vicinity = self.vicinity_instance.query_threshold(query_vector_2d, threshold=threshold)
            else:
                results_from_vicinity = self.vicinity_instance.query(query_vector_2d, k=k)
            
            if results_from_vicinity and isinstance(results_from_vicinity, list) and len(results_from_vicinity) > 0:
                actual_results_list = results_from_vicinity[0]
                return [(str(item_id), float(score)) for item_id, score in actual_results_list]
            return [] 
        except Exception as e:
            print(f"VectorStoreManager Error: Querying store failed: {e}")
            return []

    def get_store_metadata(self) -> Optional[Dict[str, Any]]:
        """Returns the metadata of the loaded store."""
        if self.vicinity_instance:
            return self.vicinity_instance.metadata
        return None

    def get_total_items(self) -> int:
        """Returns the total number of items in the store."""
        if self.vicinity_instance:
            try: return len(self.vicinity_instance)
            except TypeError: 
                print("VectorStoreManager Warning: len() not supported by Vicinity instance.")
                return 0 
        return 0