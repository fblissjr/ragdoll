# embedding_module.py
import numpy as np
from vicinity import Metric
from model2vec import StaticModel as Model2VecStaticModel
from sentence_transformers import SentenceTransformer

try:
    from sentence_transformers import SparseEncoder as SbertSparseEncoder
except ImportError:
    SbertSparseEncoder = None 

def infer_embedding_model_params(model_name: str) -> tuple[str, int, Metric, str]:
    model_name_low = model_name.lower()
    model_class_type = "dense" 

    if "potion" in model_name_low:
        return "model2vec", 8, Metric.COSINE, model_class_type
    is_known_sparse_hf_name = model_name == "tomaarsen/dual-inference-free-1e-3-lr"
    if "sparse" in model_name_low or "splade" in model_name_low or "idf" in model_name_low or is_known_sparse_hf_name:
        model_class_type = "sparse"
        dim = 30522 if is_known_sparse_hf_name else 768
        return "sentence-transformers", dim, Metric.INNER_PRODUCT, model_class_type
    if "sentence-transformers/" in model_name_low or "all-minilm" in model_name_low or \
       "msmarco" in model_name_low or "bge-" in model_name_low or "e5-" in model_name_low or \
       "gte-" in model_name_low or "instructor-" in model_name_low:
        if "large" in model_name_low: dim = 1024
        elif "base" in model_name_low: dim = 768
        elif "small" in model_name_low or "minilm" in model_name_low: dim = 384
        else: dim = 384 
        return "sentence-transformers", dim, Metric.COSINE, model_class_type
    print(f"Warning (embedding_module): Could not confidently infer parameters for model '{model_name}'. Defaulting.")
    return "sentence-transformers", 768, Metric.COSINE, "dense"

def generate_embeddings(text_list: list[str], model_name: str, model_type: str, default_dim: int, model_class_type: str = "dense", device: str | None = None) -> tuple[np.ndarray, int]:
    if not text_list: return np.empty((0, default_dim)), default_dim
    print(f"Embedding Module: Loading model: {model_name} (Type: {model_type}, Class: {model_class_type}) for device '{device}'...")
    model_instance, actual_dimension = None, default_dim
    try:
        if model_type == "model2vec":
            model_instance = Model2VecStaticModel.from_pretrained(model_name) # Model2vec might not take device arg directly here
        elif model_type == "sentence-transformers":
            model_instance = SentenceTransformer(model_name, device=device)
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")
        
        # For model2vec, if it doesn't take device in constructor, try to move after load if it's a PyTorch model
        if model_type == "model2vec" and device and hasattr(model_instance, "to"):
            try: model_instance.to(device) # This is a guess; check model2vec docs
            except Exception as e_mv_device: print(f"Embedding Module: Note - couldn't move model2vec model to {device}: {e_mv_device}")

        print(f"Embedding Module: Model '{model_name}' loaded. Effective device: {model_instance.device if hasattr(model_instance, 'device') else 'CPU (assumed)'}.")
        embeddings = model_instance.encode([str(t) for t in text_list], show_progress_bar=True, batch_size=128)
        
        if not isinstance(embeddings, np.ndarray):
            try: embeddings = embeddings.toarray()
            except AttributeError:
                try: embeddings = np.array(embeddings) 
                except: print("  Failed to cast embeddings to numpy array."); return np.empty((0, default_dim)), default_dim
        
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.ndim == 1 and vectors.size > 0 : vectors = vectors.reshape(1, -1)
        elif vectors.size == 0 and len(text_list) > 0:
            print(f"  Warning: Model '{model_name}' produced empty output."); return np.empty((0, default_dim)), default_dim
        
        if vectors.size > 0: actual_dimension = vectors.shape[1]
        else: actual_dimension = default_dim if default_dim is not None else 0 # Handle if default_dim also None
            
        if vectors.size > 0 and default_dim is not None and actual_dimension != default_dim :
            print(f"Embedding Module: Note - Actual vector dimension for '{model_name}' is {actual_dimension} (expected {default_dim}).")
        return vectors, actual_dimension
    except Exception as e:
        print(f"Embedding Module: Error with embedding model {model_name}: {e}")
        return np.empty((0, default_dim if default_dim is not None else 0)), (default_dim if default_dim is not None else 0)