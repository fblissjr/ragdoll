# reranker_module.py
from sentence_transformers import CrossEncoder
from tqdm import tqdm
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import common_utils
import os

class Reranker:
    def __init__(self, model_name: str, device: str | None = None):
        print(f"Reranker Module: Initializing CrossEncoder model '{model_name}' for device '{device}'...")
        try:
            self.model = CrossEncoder(model_name, device=device)
            print(f"Reranker Module: CrossEncoder '{model_name}' loaded. Effective device: {self.model.device}.")
        except Exception as e:
            print(f"Reranker Module: Error loading CrossEncoder model '{model_name}': {e}")
            self.model = None

    def rerank(
        self, 
        query: str, 
        chunks_with_ids: List[Tuple[str, str]], # List of (chunk_id, chunk_text)
        top_n: int,
        batch_size: int = 32
    ) -> List[Tuple[str, float]]: # Returns list of (chunk_id, reranked_score)
        if not self.model:
            print("Reranker Module: Model not loaded, cannot rerank. Returning original order (scores will be arbitrary).")
            # Return original IDs with arbitrary scores if model failed to load
            return [(chunk_id, 1.0 / (i + 1)) for i, (chunk_id, _) in enumerate(chunks_with_ids[:top_n])]

        if not query or not chunks_with_ids:
            return []

        sentence_pairs = [[query, chunk_text] for _, chunk_text in chunks_with_ids]
        
        all_scores = []
        print(f"Reranker Module: Reranking {len(sentence_pairs)} pairs with batch size {batch_size}...")
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Reranking Batches"):
            batch = sentence_pairs[i:i+batch_size]
            try:
                scores = self.model.predict(batch, convert_to_numpy=True, show_progress_bar=False) # predict is faster for batch
                all_scores.extend(scores.tolist())
            except Exception as e_predict:
                print(f"Reranker Module: Error during batch prediction: {e_predict}")
                all_scores.extend([0.0] * len(batch)) # Assign low score on error

        if not isinstance(all_scores, np.ndarray):
            all_scores = np.array(all_scores)

        # Combine IDs with their new scores
        scored_chunks = []
        for i, (chunk_id, _) in enumerate(chunks_with_ids):
            if i < len(all_scores):
                 scored_chunks.append((chunk_id, float(all_scores[i])))
            else: # Should not happen if all_scores are correctly extended
                 scored_chunks.append((chunk_id, 0.0))


        # Sort by new scores in descending order
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        return scored_chunks[:top_n]

def load_supporting_data(vector_data_dir: str) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[Dict[str, Any]]]]:
    """Loads text_chunks, chunk_ids, and detailed_metadata from JSON files."""
    # This function is identical to the one in rag_core and will be moved to a data_loader module
    # For now, keep it simple here for reranker_module's potential standalone use or testing.
    text_chunks_path = os.path.join(vector_data_dir, common_utils.TEXT_CHUNKS_FILENAME)
    chunk_ids_path = os.path.join(vector_data_dir, common_utils.CHUNK_IDS_FILENAME)
    detailed_metadata_path = os.path.join(vector_data_dir, common_utils.DETAILED_CHUNK_METADATA_FILENAME)

    if not all(os.path.exists(p) for p in [text_chunks_path, chunk_ids_path, detailed_metadata_path]):
        return None, None, None
    try:
        with open(text_chunks_path, 'r', encoding='utf-8') as f: tc = json.load(f)
        with open(chunk_ids_path, 'r', encoding='utf-8') as f: ci = json.load(f)
        with open(detailed_metadata_path, 'r', encoding='utf-8') as f: dm = json.load(f)
        return tc, ci, dm
    except Exception as e:
        print(f"Reranker (load_supporting_data): Error loading data: {e}"); return None, None, None