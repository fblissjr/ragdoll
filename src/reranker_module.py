# src/reranker_module.py
# Provides a Reranker class using a CrossEncoder model to re-score retrieved chunks.

from sentence_transformers import CrossEncoder 
from tqdm import tqdm 
import numpy as np
from typing import List, Tuple, Optional 
# No specific imports needed from ragdoll_config or ragdoll_utils for core reranking logic,
# as model names and batch sizes are passed or are defaults within the class/methods.

class Reranker:
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initializes the Reranker with a CrossEncoder model.
        """
        print(f"Reranker Module: Initializing CrossEncoder '{model_name}' for device '{device}'...")
        self.model: Optional[CrossEncoder] = None 
        try:
            self.model = CrossEncoder(model_name, device=device)
            effective_device = str(self.model.device if hasattr(self.model, 'device') else "Unknown")
            print(f"Reranker Module: Model '{model_name}' loaded. Effective device: {effective_device}.")
        except Exception as e:
            print(f"Reranker Module Error: Loading CrossEncoder '{model_name}': {e}")

    def rerank(
        self, 
        query: str, 
        chunks_with_ids: List[Tuple[str, str]], 
        top_n: int, 
        batch_size: int = 32 # Default batch size for prediction
    ) -> List[Tuple[str, float]]:
        """
        Reranks chunks based on relevance to the query.
        """
        if not self.model:
            print("Reranker Module Warning: Model not loaded. Cannot rerank. Returning empty list.")
            return []
        if not query or not chunks_with_ids:
            print("Reranker Module Info: Empty query or chunks provided. Returning empty list.")
            return []

        sentence_pairs = [[query, chunk_text] for _, chunk_text in chunks_with_ids]
        all_scores_list: List[float] = []
        
        print(f"Reranker Module: Reranking {len(sentence_pairs)} pairs (Batch size: {batch_size})...")
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Reranking Batches", disable=len(sentence_pairs) < batch_size*2):
            batch = sentence_pairs[i : i + batch_size]
            try:
                scores = self.model.predict(batch, convert_to_numpy=True, show_progress_bar=False)
                all_scores_list.extend(scores.tolist())
            except Exception as e_predict:
                print(f"Reranker Module Error: Batch prediction failed: {e_predict}")
                all_scores_list.extend([0.0] * len(batch)) # Assign low score on error

        # Combine IDs with their new scores
        scored_chunks: List[Tuple[str, float]] = []
        for i, (chunk_id, _) in enumerate(chunks_with_ids):
            if i < len(all_scores_list):
                 scored_chunks.append((chunk_id, float(all_scores_list[i])))
            else: 
                 print(f"Reranker Module Warning: Score missing for chunk_id {chunk_id}. Assigning 0.0.")
                 scored_chunks.append((chunk_id, 0.0))

        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return scored_chunks[:top_n]