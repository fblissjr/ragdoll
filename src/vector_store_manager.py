# vector_store_manager.py
import os
import shutil
import json
import numpy as np
from datetime import datetime
from vicinity import Vicinity, Backend, Metric # Make sure vicinity is installed

import common_utils

class VectorStoreManager:
    def __init__(self, vector_data_dir: str, vicinity_store_name: str):
        self.vector_data_dir = vector_data_dir
        self.vicinity_store_name = vicinity_store_name
        self.vicinity_store_path = os.path.join(self.vector_data_dir, self.vicinity_store_name)
        self.vicinity_instance: Vicinity | None = None
        self._ensure_data_dir_exists()

    def _ensure_data_dir_exists(self):
        os.makedirs(self.vector_data_dir, exist_ok=True)

    def create_and_save_store(
        self, 
        vectors: np.ndarray, 
        items: list[str], 
        metric: Metric, 
        metadata_dict: dict, 
        overwrite: bool = False
    ) -> bool:
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2 or vectors.shape[0] == 0:
            print("VectorStoreManager Error: Invalid or empty vectors provided.")
            return False
        if not items or len(vectors) != len(items):
            print("VectorStoreManager Error: Items list is empty or does not match vector count.")
            return False
        
        if os.path.exists(self.vicinity_store_path):
            if overwrite:
                print(f"VectorStoreManager: Overwriting existing store at '{self.vicinity_store_path}'...")
                if os.path.isdir(self.vicinity_store_path):
                    shutil.rmtree(self.vicinity_store_path)
                else:
                    os.remove(self.vicinity_store_path) # Should be a dir, but handle if it's a file
            else:
                print(f"VectorStoreManager Error: Store already exists at '{self.vicinity_store_path}' and overwrite is False.")
                return False
        
        try:
            print("VectorStoreManager: Initializing Vicinity store...")
            self.vicinity_instance = Vicinity.from_vectors_and_items(
                vectors=vectors,
                items=items,
                backend_type=Backend.BASIC, # Or other backends if Vicinity supports them and they are configurable
                metric=metric
            )
            self.vicinity_instance.metadata = metadata_dict
            print(f"VectorStoreManager: Saving Vicinity store to '{self.vicinity_store_path}'...")
            self.vicinity_instance.save(self.vicinity_store_path)
            print("VectorStoreManager: Vicinity store saved successfully.")
            return True
        except Exception as e:
            print(f"VectorStoreManager Error: Failed to create/save Vicinity store: {e}")
            self.vicinity_instance = None
            return False

    def load_store(self) -> bool:
        if not os.path.exists(self.vicinity_store_path):
            print(f"VectorStoreManager Error: Store not found at '{self.vicinity_store_path}'. Cannot load.")
            return False
        try:
            print(f"VectorStoreManager: Loading Vicinity store from '{self.vicinity_store_path}'...")
            self.vicinity_instance = Vicinity.load(self.vicinity_store_path)
            print("VectorStoreManager: Vicinity store loaded successfully.")
            if self.vicinity_instance and self.vicinity_instance.metadata:
                print("  Store Metadata:")
                for key, value in self.vicinity_instance.metadata.items():
                    if isinstance(value, dict) or isinstance(value, list) and len(value) > 5:
                        print(f"    {key}: (complex type or long list, preview if dict: {list(value.keys()) if isinstance(value, dict) else str(value)[:100]})")
                    else:
                        print(f"    {key}: {value}")
            return True
        except Exception as e:
            print(f"VectorStoreManager Error: Failed to load Vicinity store: {e}")
            self.vicinity_instance = None
            return False

    def query_store(self, query_vector: np.ndarray, k: int = 5, threshold: float | None = None) -> list[tuple[str, float]]:
        if self.vicinity_instance is None:
            print("VectorStoreManager Error: Store not loaded. Cannot query.")
            return []
        if not isinstance(query_vector, np.ndarray) or query_vector.ndim != 1:
            print("VectorStoreManager Error: Query vector must be a 1D NumPy array.")
            return []
        
        try:
            if threshold is not None:
                # Vicinity's query_threshold returns a list of lists, even for a single query vector
                results_wrapper = self.vicinity_instance.query_threshold(query_vector.flatten(), threshold=threshold)
            else:
                # Vicinity's query returns a list of lists
                results_wrapper = self.vicinity_instance.query(query_vector.flatten(), k=k)
            
            # Assuming results_wrapper is like [[(item1, score1), (item2, score2)]] for a single query vector
            if results_wrapper and isinstance(results_wrapper, list) and isinstance(results_wrapper[0], list):
                actual_results = results_wrapper[0]
                return [(str(item_id), float(score)) for item_id, score in actual_results]
            elif results_wrapper and isinstance(results_wrapper, list) and all(isinstance(el, tuple) for el in results_wrapper):
                 # This might happen if query_threshold for a single vector doesn't wrap result in another list.
                 # Or if .query for k=1 also doesn't wrap. Let's assume this structure could happen.
                return [(str(item_id), float(score)) for item_id, score in results_wrapper]

            return []
        except Exception as e:
            print(f"VectorStoreManager Error: Failed to query store: {e}")
            return []

    def get_store_metadata(self) -> dict | None:
        if self.vicinity_instance:
            return self.vicinity_instance.metadata
        return None

    def get_total_items(self) -> int:
        if self.vicinity_instance:
            return len(self.vicinity_instance) # Vicinity stores should support len()
        return 0