# src/embedding_module.py
# This module is responsible for loading embedding models and generating vector embeddings for text.

import numpy as np
from vicinity import Metric 
from model2vec import StaticModel as Model2VecStaticModel 
from sentence_transformers import SentenceTransformer 
import torch 
import os # For path checking
import json # For loading config.json

try:
    from sentence_transformers import SparseEncoder as SbertSparseEncoder
except ImportError:
    SbertSparseEncoder = None 

def _try_load_dimension_from_local_config(model_path: str) -> Optional[int]:
    """Helper to load dimension from a model's config.json if it's a local path."""
    if os.path.isdir(model_path):
        config_path = os.path.join(model_path, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                # Common keys for dimension in Hugging Face model configs
                dim_keys = ["hidden_size", "dim", "d_model", "dimension", "out_features"]
                for key in dim_keys:
                    if key in config_data and isinstance(config_data[key], int):
                        print(f"Embedding Module Info: Inferred dimension {config_data[key]} from '{key}' in {config_path} for local model.")
                        return config_data[key]
                # Specific to Model2Vec's StaticModel config if different
                if "projection_dims" in config_data and isinstance(config_data["projection_dims"], int): # Model2Vec Potion often has this
                     print(f"Embedding Module Info: Inferred dimension {config_data['projection_dims']} from 'projection_dims' in {config_path} for local model.")
                     return config_data["projection_dims"]
                if "embed_dim" in config_data and isinstance(config_data["embed_dim"], int): # Another key sometimes used
                     print(f"Embedding Module Info: Inferred dimension {config_data['embed_dim']} from 'embed_dim' in {config_path} for local model.")
                     return config_data["embed_dim"]

            except Exception as e:
                print(f"Embedding Module Warning: Could not read dimension from local config.json at {config_path}: {e}")
    return None


def infer_embedding_model_params(model_name: str) -> tuple[str, int, Metric, str]:
    """
    Infers embedding model parameters (type, dimension, metric, class) based on the model name or path.
    """
    model_name_low = model_name.lower()
    model_class_type = "dense"
    default_dimension = 768 # A general fallback dimension

    # Attempt to infer dimension if model_name is a local path to a model directory
    local_dim = _try_load_dimension_from_local_config(model_name)
    if local_dim is not None:
        default_dimension = local_dim # Use dimension from local config if found

    # Heuristics for Model2Vec Potion models (name or path component)
    # Local Tokenlearn'd models might be saved in paths like "models/my_potion_model"
    if "potion" in model_name_low or (os.path.isdir(model_name) and "vectors.npy" in os.listdir(model_name) and "tokenizer.json" in os.listdir(model_name)):
        # If local_dim was found, it's used. Otherwise, Potion models often have smaller dimensions.
        # If local_dim is None here, it means we couldn't read from config, so use a Potion-like default.
        potion_default_dim = local_dim if local_dim is not None else 384 
        return "model2vec", potion_default_dim, Metric.COSINE, model_class_type 

    is_known_sparse_hf_name = "splade" in model_name_low or \
                              model_name == "tomaarsen/dual-inference-free-1e-3-lr"
    if "sparse" in model_name_low or "idf" in model_name_low or is_known_sparse_hf_name:
        model_class_type = "sparse"
        dim = local_dim if local_dim is not None else (30522 if "splade" in model_name_low and "distil" in model_name_low else 768)
        return "sentence-transformers", dim, Metric.INNER_PRODUCT, model_class_type

    sbert_indicators = ["sentence-transformers/", "all-minilm", "msmarco", "bge-", "e5-", "gte-", "instructor-", "stella", "jina", "voyage"]
    if any(indicator in model_name_low for indicator in sbert_indicators):
        sbert_dim = default_dimension # Start with potentially locally inferred dim
        if local_dim is None: # Only apply SBERT name heuristics if local inference failed
            if "large" in model_name_low: sbert_dim = 1024
            elif "base" in model_name_low: sbert_dim = 768
            elif "small" in model_name_low or "mini" in model_name_low or "minilm" in model_name_low: sbert_dim = 384
            elif "tiny" in model_name_low: sbert_dim = 384 
            else: sbert_dim = 768 
        return "sentence-transformers", sbert_dim, Metric.COSINE, model_class_type
    
    print(f"Embedding Module Warning: Could not confidently infer parameters for model/path '{model_name}'. "
          f"Using fallback: sentence-transformers, {default_dimension} dims, COSINE, dense.")
    return "sentence-transformers", default_dimension, Metric.COSINE, "dense"


def generate_embeddings(
    text_list: list[str], 
    model_name: str, 
    model_type: str, 
    default_inferred_dim: int, 
    model_class_type: str = "dense", 
    device: Optional[str] = None 
) -> tuple[np.ndarray, int]:
    """
    Generates vector embeddings for a list of text strings using the specified model.
    (Function body remains the same as in the previous update - no changes needed here based on infer_params refinement)
    """
    if not text_list: 
        dim_to_return = default_inferred_dim if default_inferred_dim is not None else 0
        return np.empty((0, dim_to_return)), dim_to_return
    
    print(f"Embedding Module: Loading model '{model_name}' (Type: {model_type}, Class: {model_class_type}) for device '{device}'...")
    model_instance = None
    actual_embedding_dimension = default_inferred_dim 
    
    try:
        if model_type == "model2vec":
            model_instance = Model2VecStaticModel.from_pretrained(model_name) 
            if device and hasattr(model_instance, "to") and callable(model_instance.to):
                try: 
                    model_instance.to(device)
                except Exception as e_mv_device: 
                    print(f"Embedding Module Warning: Could not move model2vec model '{model_name}' to device '{device}': {e_mv_device}")
        elif model_type == "sentence-transformers":
            model_instance = SentenceTransformer(model_name, device=device)
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
        
        effective_device_str = "CPU (assumed)"
        if hasattr(model_instance, 'device'): 
            dev_attr = model_instance.device
            effective_device_str = str(dev_attr) if isinstance(dev_attr, torch.device) else str(dev_attr)
        elif hasattr(model_instance, "_target_device"): 
             dev_attr = model_instance._target_device
             effective_device_str = str(dev_attr) if isinstance(dev_attr, torch.device) else str(dev_attr)

        print(f"Embedding Module: Model '{model_name}' loaded. Effective device: {effective_device_str}.")
        texts_to_encode = [str(t) if t is not None else "" for t in text_list]
        embeddings_output = model_instance.encode(texts_to_encode, show_progress_bar=len(texts_to_encode) > 200, batch_size=128)
        
        if not isinstance(embeddings_output, np.ndarray):
            try: 
                if hasattr(embeddings_output, 'toarray'): embeddings_output = embeddings_output.toarray() 
                else: embeddings_output = np.array(embeddings_output) 
            except Exception as e_cast: 
                print(f"  Embedding Module Error: Casting embeddings to NumPy array failed: {e_cast}. Type: {type(embeddings_output)}."); 
                return np.empty((0, default_inferred_dim)), default_inferred_dim
        
        vectors_np_array = np.array(embeddings_output, dtype=np.float32) 
        
        if vectors_np_array.ndim == 1 and vectors_np_array.size > 0 : 
            vectors_np_array = vectors_np_array.reshape(1, -1) 
        elif vectors_np_array.size == 0 and len(texts_to_encode) > 0:
            print(f"  Embedding Module Warning: Model '{model_name}' produced empty output for non-empty input."); 
            return np.empty((0, default_inferred_dim)), default_inferred_dim
        
        if vectors_np_array.size > 0: 
            actual_embedding_dimension = vectors_np_array.shape[1]
        
        if vectors_np_array.size > 0 and default_inferred_dim is not None and \
           actual_embedding_dimension != default_inferred_dim :
            print(f"Embedding Module Info: Actual vector dimension for '{model_name}' is {actual_embedding_dimension} "
                  f"(inferred/expected {default_inferred_dim}). Using actual dimension.")
        
        return vectors_np_array, actual_embedding_dimension
        
    except Exception as e: 
        print(f"Embedding Module Error: Exception during embedding generation with '{model_name}': {e}")
        return np.empty((0, default_inferred_dim)), default_inferred_dim