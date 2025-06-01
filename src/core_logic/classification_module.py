# src/core_logic/classification_module.py
# This module handles the zero-shot classification of text chunks.

import torch
from tqdm import tqdm
from typing import List, Dict, Optional, Any

# Third-party library for Hugging Face pipelines
from transformers import pipeline as hf_pipeline 
from transformers.pipelines.base import Pipeline as HFPipeline # For type hinting

# Project-specific config for default labels
from .. import ragdoll_config 


def initialize_classifier(model_name: str, device_id: int = -1) -> Optional[HFPipeline]: 
    """
    Initializes and returns a Hugging Face zero-shot classification pipeline.
    Manages device placement (CPU, CUDA, MPS for Apple Silicon).
    """
    try: 
        effective_device_idx: Union[int, str] = -1 # Default to CPU
        
        if device_id >= 0: # If a specific GPU device ID is requested
            if torch.cuda.is_available():
                effective_device_idx = device_id 
                print(f"Classification Module: Initializing Classifier '{model_name}' on CUDA device: cuda:{effective_device_idx}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): # Check for Apple Silicon MPS
                effective_device_idx = "mps" # Use string "mps" for pipeline's device argument
                print(f"Classification Module: Initializing Classifier '{model_name}' on MPS device.")
            else: # Requested GPU not available, fallback to CPU
                cpu_reason = f"Selected GPU device_id {device_id} not available"
                print(f"Classification Module: {cpu_reason}. Initializing Classifier '{model_name}' on CPU.")
        else: # Explicitly selected CPU (device_id < 0)
            print(f"Classification Module: Initializing Classifier '{model_name}' on CPU (device_id: {device_id}).")
            
        # Load the zero-shot-classification pipeline from Hugging Face.
        classifier = hf_pipeline(
            task="zero-shot-classification", 
            model=model_name, 
            device=effective_device_idx # Pass the determined device index or string
        )
        print(f"Classification Module: Classifier '{model_name}' loaded successfully on device: {classifier.device}.")
        return classifier
    except Exception as e_clf: # Catch any error during model loading or pipeline creation
        print(f"Classification Module Error: Failed to initialize classifier '{model_name}': {e_clf}. Classification will be skipped.")
        return None

def classify_chunks_batch(
    chunks_to_classify: list[str], 
    classifier_pipeline: HFPipeline, 
    candidate_labels: list[str], 
    classification_batch_size: int, 
    progress_desc: str = "Classifying Chunks"
) -> list[Optional[dict]]: 
    """
    Classifies a list of text chunks in batches using the provided zero-shot classifier pipeline.
    Returns a list of dictionaries, each containing classification results for a corresponding chunk.
    """
    # Basic validation for inputs
    if classifier_pipeline is None or not chunks_to_classify: 
        # If no classifier or no chunks, return a list of statuses indicating skipped classification
        return [{"classification_status": "skipped_no_classifier_or_chunks"}] * len(chunks_to_classify)
    if not candidate_labels: 
        print("Classification Module: No candidate labels provided for classification. Skipping.")
        return [{"classification_status": "skipped_no_labels"}] * len(chunks_to_classify)
    
    all_classification_outputs = [] # To store results for all chunks
    try:
        # Process chunks in batches for better performance and memory management
        for i in tqdm(range(0, len(chunks_to_classify), classification_batch_size), 
                      desc=progress_desc, 
                      disable=len(chunks_to_classify) < classification_batch_size * 2): # Disable progress bar for very small jobs
            
            current_batch_texts = chunks_to_classify[i : i + classification_batch_size]
            if not current_batch_texts: continue # Should not happen with correct loop logic

            # Perform zero-shot classification on the current batch.
            # 'multi_label=False' implies we are interested in the single best label per chunk.
            hf_pipeline_outputs = classifier_pipeline(current_batch_texts, candidate_labels=candidate_labels, multi_label=False) 
            
            # The pipeline might return a single dict if the batch size was 1. Standardize to a list.
            if isinstance(hf_pipeline_outputs, dict) and len(current_batch_texts) == 1: 
                hf_pipeline_outputs = [hf_pipeline_outputs]
            elif not isinstance(hf_pipeline_outputs, list): 
                print(f"  Classification Warning: Unexpected output type from classifier: {type(hf_pipeline_outputs)}. Expected list for batch. Batch size: {len(current_batch_texts)}")
                # Add error status for each item in this problematic batch
                all_classification_outputs.extend([{"classification_error": "unexpected_classifier_output_type"}] * len(current_batch_texts))
                continue # Skip to the next batch
            
            # Process each result item from the Hugging Face pipeline output
            for result_item_dict in hf_pipeline_outputs: 
                all_classification_outputs.append({
                    "top_label": result_item_dict["labels"][0] if result_item_dict.get("labels") and result_item_dict["labels"] else "N/A", 
                    "top_label_score": float(result_item_dict["scores"][0]) if result_item_dict.get("scores") and result_item_dict["scores"] else 0.0, 
                    "zero_shot_labels": result_item_dict.get("labels", []), # Full list of labels returned by HF (often all candidates)
                    "zero_shot_scores": [float(s) for s in result_item_dict.get("scores", [])] # Corresponding scores
                })
        return all_classification_outputs
    except Exception as e_clf_batch: 
        print(f"Classification Module Error: Exception during batch classification: {e_clf_batch}")
        # If a batch-level error occurs, return an error status for all chunks
        return [{"classification_error": str(e_clf_batch)}] * len(chunks_to_classify)