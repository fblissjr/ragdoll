# src/embedding_module.py
# This module is responsible for loading embedding models and generating vector embeddings for text.
import numpy as np
from vicinity import Metric 
from sentence_transformers import SentenceTransformer
import torch
import os # Ensure os is imported for os.path.isdir
from pathlib import Path # Ensure Path is imported
import json
from typing import Optional, Tuple, List, Any

# Model2Vec (if used and installed)
MODEL2VEC_AVAILABLE = False
Model2VecStaticModel: Optional[type] = None
try:
    from model2vec import StaticModel as ImportedModel2VecStaticModel
    Model2VecStaticModel = ImportedModel2VecStaticModel
    MODEL2VEC_AVAILABLE = True
    print("Embedding Module: Model2Vec loaded successfully.")
except ImportError:
    print("Embedding Module: Model2Vec not found. Potion models might not be usable via Model2Vec directly.")


def _try_load_dimension_from_local_config(model_path_str: str) -> Optional[int]:
    """
    Attempts to load 'embedding_dimension' or 'hidden_size' from a 'config.json' 
    in the given local model directory.
    """
    config_path = Path(model_path_str) / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            dim_keys = ["embedding_dimension", "hidden_size", "d_model"] # Common keys for dimension
            for key in dim_keys:
                dim = config_data.get(key)
                if isinstance(dim, int):
                    print(f"  Embedding Module Info: Loaded dimension {dim} from local config.json (key: '{key}') for '{model_path_str}'")
                    return dim
            print(f"  Embedding Module Info: No standard dimension key found in config.json for '{model_path_str}'")
        except Exception as e_config:
            print(f"  Embedding Module Warning: Could not read/parse config.json for dimension at '{model_path_str}': {e_config}")
    return None

def infer_embedding_model_params(
    model_name_or_path: str
) -> Tuple[str, int, Metric, str]: # Returns: (type, dimension, metric_enum, class_type)
    """
    Infers embedding model type, default dimension, Vicinity metric, and model class (dense/sparse)
    based on the model name or path.
    Tries to load dimension from local config.json if model_name_or_path is a directory.
    """
    model_name_low = model_name_or_path.lower()
    
    # Initialize with safe defaults
    inferred_model_type_str = "sentence-transformers" 
    inferred_dimension_val = 384 # A common small SBERT dimension
    inferred_metric_enum_val = Metric.COSINE
    inferred_model_class_str = "dense"

    local_dim_from_config: Optional[int] = None
    if os.path.isdir(model_name_or_path): # Check if it's a local directory path
        local_dim_from_config = _try_load_dimension_from_local_config(model_name_or_path)
        if local_dim_from_config is not None:
            inferred_dimension_val = local_dim_from_config # Prioritize dimension from local config

    # Heuristics based on model name patterns
    if MODEL2VEC_AVAILABLE and ("potion" in model_name_low or \
                                (Model2VecStaticModel and Model2VecStaticModel.is_static_model(model_name_or_path))):
        inferred_model_type_str = "model2vec"
        inferred_dimension_val = local_dim_from_config if local_dim_from_config is not None else 256 
        inferred_metric_enum_val = Metric.COSINE 
        inferred_model_class_str = "dense"
    elif "splade" in model_name_low or "sparse" in model_name_low: # Check for sparse model indicators
        inferred_model_type_str = "sentence-transformers" 
        inferred_model_class_str = "sparse"
        inferred_dimension_val = local_dim_from_config if local_dim_from_config is not None else 30522 # Common for SPLADE vocab size
        inferred_metric_enum_val = Metric.INNER_PRODUCT
    elif any(st_indicator in model_name_low for st_indicator in 
             ["all-", "msmarco", "bge-", "e5-", "gte-", "instructor-", "sentence-transformers/"]):
        inferred_model_type_str = "sentence-transformers"
        if local_dim_from_config is not None: # Already set if found
            pass 
        elif "large" in model_name_low: inferred_dimension_val = 1024
        elif "base" in model_name_low: inferred_dimension_val = 768
        elif "small" in model_name_low or "mini" in model_name_low: inferred_dimension_val = 384
        else: inferred_dimension_val = 384 # Fallback SBERT dimension if not specified by size
        inferred_metric_enum_val = Metric.COSINE
        inferred_model_class_str = "dense"
    else: 
        # If not clearly identifiable by name, assume SBERT and use dimension from local config if found,
        # or the initial default.
        inferred_model_type_str = "sentence-transformers"
        # inferred_dimension_val is already set (either from local_config or initial default)
        inferred_metric_enum_val = Metric.COSINE
        inferred_model_class_str = "dense"
        print(f"Embedding Module Warning: Could not confidently infer specific parameters for model '{model_name_or_path}' by name pattern. "
              f"Using: type='{inferred_model_type_str}', dim={inferred_dimension_val}, metric='{inferred_metric_enum_val}', class='{inferred_model_class_str}'. "
              "Consider providing explicit overrides if these are incorrect.")

    return inferred_model_type_str, inferred_dimension_val, inferred_metric_enum_val, inferred_model_class_str

def generate_embeddings(
    text_list: List[str], 
    model_name_or_path: str,             
    # User overrides (can be None if not provided by user)
    model_type_override: Optional[str],  
    expected_dimension_override: Optional[int], 
    # Parameters inferred by infer_embedding_model_params:
    inferred_model_type: str,       # The model type string ("model2vec", "sentence-transformers")
    inferred_model_class: str,      # "dense" or "sparse"
    inferred_dimension: int,        # Dimension inferred from name/local model config
    device: Optional[str] = None    
) -> Tuple[np.ndarray, int]:
    """
    Generates embeddings for a list of texts.
    Uses inferred parameters but allows user overrides for type and dimension.
    The actual dimension of the generated embeddings is returned.
    """
    if not text_list:
        # Use the most reliable dimension available for shaping the empty array
        final_dim_for_empty_array = expected_dimension_override if expected_dimension_override is not None else inferred_dimension
        return np.empty((0, final_dim_for_empty_array)), final_dim_for_empty_array

    # Determine final model type and expected dimension to use for loading
    actual_model_type_to_load = model_type_override or inferred_model_type
    # Prioritize user override for dimension, then inferred, then a fallback.
    current_expected_dimension_for_loading = expected_dimension_override if expected_dimension_override is not None \
        else inferred_dimension

    print(f"Embedding Module: Loading model '{model_name_or_path}' "
          f"(Type to load: {actual_model_type_to_load}, Class: {inferred_model_class}, Expected Dim: {current_expected_dimension_for_loading}) "
          f"for device '{device}'...")

    model_instance: Any = None 
    actual_generated_dimension = current_expected_dimension_for_loading # Initialize, will be updated

    try:
        if actual_model_type_to_load == "model2vec" and MODEL2VEC_AVAILABLE and Model2VecStaticModel:
            model_instance = Model2VecStaticModel.from_pretrained(model_name_or_path)
            print(f"Embedding Module: Model2Vec model '{model_name_or_path}' loaded. Device management is internal to Model2Vec.")
        elif actual_model_type_to_load == "sentence-transformers":
            model_instance = SentenceTransformer(model_name_or_path, device=device)
            effective_sbert_device = str(model_instance.device) if hasattr(model_instance, 'device') else "Unknown (SBERT)"
            print(f"Embedding Module: SentenceTransformer model '{model_name_or_path}' loaded. Effective device: {effective_sbert_device}.")
        else:
            raise ValueError(f"Unsupported or unavailable embedding model type: '{actual_model_type_to_load}'. "
                             f"Ensure type is 'model2vec' (and installed) or 'sentence-transformers'.")
        
        encode_kwargs = {"show_progress_bar": True, "batch_size": 128} 
        if actual_model_type_to_load == "model2vec":
            encode_kwargs.pop("show_progress_bar", None) 
        
        embeddings_raw = model_instance.encode([str(t) for t in text_list], **encode_kwargs)
        
        if not isinstance(embeddings_raw, np.ndarray):
            try: 
                if inferred_model_class == "sparse" and isinstance(embeddings_raw, list) and embeddings_raw and isinstance(embeddings_raw[0], dict):
                    print("Embedding Module: Sparse embeddings detected. Constructing dense representation (placeholder).")
                    num_features = current_expected_dimension_for_loading 
                    dense_vectors = []
                    for sparse_dict_item in embeddings_raw:
                        vec = np.zeros(num_features, dtype=np.float32)
                        for idx, val in sparse_dict_item.items():
                            if isinstance(idx, int) and idx < num_features: vec[idx] = val
                        dense_vectors.append(vec)
                    embeddings_np = np.array(dense_vectors, dtype=np.float32)
                else:
                    embeddings_np = np.array(embeddings_raw, dtype=np.float32)
            except Exception as e_cast:
                print(f"  Embedding Module Error: Failed to cast embeddings to NumPy array: {e_cast}. Type was: {type(embeddings_raw)}")
                return np.empty((0, current_expected_dimension_for_loading)), current_expected_dimension_for_loading
        else:
            embeddings_np = embeddings_raw.astype(np.float32)

        if embeddings_np.ndim == 1 and embeddings_np.size > 0 : 
            embeddings_np = embeddings_np.reshape(1, -1) 
        elif embeddings_np.size == 0 and len(text_list) > 0:
            print(f"  Embedding Module Warning: Model '{model_name_or_path}' produced empty output for non-empty input list.")
            return np.empty((0, current_expected_dimension_for_loading)), current_expected_dimension_for_loading
        
        if embeddings_np.size > 0:
            actual_generated_dimension = embeddings_np.shape[1]
            if actual_generated_dimension != current_expected_dimension_for_loading:
                print(f"Embedding Module Info: Actual vector dimension for '{model_name_or_path}' is {actual_generated_dimension} "
                      f"(expected/inferred was {current_expected_dimension_for_loading}). Using actual dimension.")
        else: 
            actual_generated_dimension = current_expected_dimension_for_loading
            
        return embeddings_np, actual_generated_dimension

    except Exception as e_embed_gen:
        print(f"Embedding Module Error: Failed during embedding generation with model '{model_name_or_path}': {e_embed_gen}")
        return np.empty((0, current_expected_dimension_for_loading)), current_expected_dimension_for_loading