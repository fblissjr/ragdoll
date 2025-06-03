# src/core_logic/classification_module.py
import torch
from tqdm import tqdm # Ensure tqdm is imported
from typing import List, Dict, Any, Optional, Iterator
from transformers import pipeline as hf_pipeline, Pipeline 
from datasets import Dataset # Keep this import if you want to try it later

# RAGdoll project-specific imports (if any are needed for defaults)
from .. import ragdoll_config 

def initialize_classifier(
    model_name_or_path: str, 
    device_id: int = -1, 
    task: str = "zero-shot-classification",
    # This batch_size is passed to the hf_pipeline constructor.
    # It tells the pipeline how to configure its internal DataLoader.
    batch_size_for_pipeline_init: int = ragdoll_config.DEFAULT_CLASSIFICATION_BATCH_SIZE 
) -> Optional[Pipeline]:
    """
    Initializes and returns a Hugging Face zero-shot classification pipeline.
    The pipeline is configured with the specified batch_size_for_pipeline_init.
    """
    try:
        effective_device = -1 
        if device_id >= 0:
            if torch.cuda.is_available():
                effective_device = device_id 
                print(f"Classification Module: Attempting to use CUDA device: cuda:{effective_device}")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
                effective_device = "mps" 
                print(f"Classification Module: Attempting to use MPS device (Apple Silicon).")
            else:
                print(f"Classification Module: GPU device_id {device_id} requested, but CUDA/MPS not available. Using CPU.")
        else:
            print(f"Classification Module: Using CPU for classifier.")
            
        print(f"Classification Module: Initializing Classifier '{model_name_or_path}' on device: {effective_device} with pipeline init batch_size: {batch_size_for_pipeline_init}")
        
        classifier_pipeline = hf_pipeline(
            task=task, 
            model=model_name_or_path, 
            device=effective_device,
            batch_size=batch_size_for_pipeline_init # Pass batch_size during pipeline initialization
        )
        
        actual_pipeline_device = "unknown"
        if hasattr(classifier_pipeline, 'model') and hasattr(classifier_pipeline.model, 'device'):
            actual_pipeline_device = str(classifier_pipeline.model.device)
        elif hasattr(classifier_pipeline, 'device'):
             actual_pipeline_device = str(classifier_pipeline.device)

        # We print the batch_size it was initialized with for confirmation.
        # The actual attribute name might vary or not be public.
        init_bs_msg = f"Pipeline configured init batch_size: {batch_size_for_pipeline_init}"
        if hasattr(classifier_pipeline, 'batch_size') and classifier_pipeline.batch_size is not None:
            init_bs_msg = f"Pipeline public batch_size attribute: {classifier_pipeline.batch_size}"
        elif hasattr(classifier_pipeline, '_batch_size') and classifier_pipeline._batch_size is not None: # Check private common name
             init_bs_msg = f"Pipeline internal _batch_size attribute: {classifier_pipeline._batch_size}"


        print(f"Classification Module: Classifier '{model_name_or_path}' loaded. "
              f"Effective model device: {actual_pipeline_device}. {init_bs_msg}")
        return classifier_pipeline
    except Exception as e_classifier_init:
        print(f"Classification Module Error: Failed to initialize classifier '{model_name_or_path}': {e_classifier_init}")
        return None

def classify_chunks_batch(
    chunks_to_classify: List[str], 
    classifier: Pipeline, 
    candidate_labels: List[str], 
    # This batch_size is passed to the pipeline's __call__ method.
    # It can potentially override or inform the batching for this specific call.
    batch_size_for_call: int = ragdoll_config.DEFAULT_CLASSIFICATION_BATCH_SIZE,
    multi_label_classification: bool = False
) -> List[Optional[Dict[str, Any]]]:
    """
    Classifies a list of text chunks using a pre-initialized zero-shot classifier.
    The pipeline object handles internal batching. We can also suggest a batch_size for the call.
    """
    if classifier is None or not hasattr(classifier, '__call__'):
        print("Classification Module: Classifier not initialized or invalid. Skipping.")
        return [{"classification_status": "skipped_no_classifier"}] * len(chunks_to_classify)
    if not chunks_to_classify: 
        print("Classification Module: No chunks to classify.")
        return []
    if not candidate_labels:
        print("Classification Module: No candidate labels for classification. Skipping.")
        return [{"classification_status": "skipped_no_labels"}] * len(chunks_to_classify)

    all_classification_results: List[Optional[Dict[str, Any]]] = []
    
    num_chunks = len(chunks_to_classify)
    print(f"Classification Module: Starting classification for {num_chunks} chunks.")
    # Log the batch_size the pipeline was initialized with (if accessible) and the one for the call.
    init_bs_msg = "N/A (not exposed)"
    if hasattr(classifier, 'batch_size') and classifier.batch_size is not None: init_bs_msg = str(classifier.batch_size)
    elif hasattr(classifier, '_batch_size') and classifier._batch_size is not None: init_bs_msg = str(classifier._batch_size)
    print(f"  Pipeline init batch_size: {init_bs_msg}. Batch_size for this call: {batch_size_for_call}.")

    try:
        # Option A: Pass List[str] directly + batch_size kwarg to the call
        # This is often sufficient and simpler if the pipeline supports it well.
        pipeline_outputs_iterator = classifier(
            chunks_to_classify, # Pass List[str] directly
            candidate_labels=candidate_labels, 
            multi_label=multi_label_classification,
            batch_size=batch_size_for_call # Explicitly pass batch_size to the call
        )
        
        # Option B: If the "use a dataset" warning persists with Option A, use datasets.Dataset
        # from datasets import Dataset 
        # hf_dataset = Dataset.from_list([{"text": item} for item in chunks_to_classify])
        # print(f"Classification Module: Using Hugging Face Dataset for processing.")
        # pipeline_outputs_iterator = classifier(
        #     hf_dataset,
        #     candidate_labels=candidate_labels, 
        #     multi_label=multi_label_classification
        #     # batch_size for the call might also be passed here if using Dataset,
        #     # but often the pipeline's init batch_size is primary when Dataset is input.
        # )
        
        # tqdm will iterate over the results as the pipeline yields them.
        # The pipeline processes internally in batches, then yields results.
        # The progress bar will update per *result item*, not per batch processed internally by the pipeline.
        for i, result_item in enumerate(tqdm(pipeline_outputs_iterator, total=num_chunks, desc="Classifying Chunks")):
            if isinstance(result_item, dict) and "labels" in result_item and "scores" in result_item:
                all_classification_results.append({
                    "top_label": result_item["labels"][0] if result_item["labels"] else "N/A",
                    "top_label_score": float(result_item["scores"][0]) if result_item["scores"] else 0.0,
                    "zero_shot_labels": result_item.get("labels", []),
                    "zero_shot_scores": [float(s) for s in result_item.get("scores", [])],
                    "classification_status": "success"
                })
            else:
                status_note = "error_unexpected_pipeline_output_format"
                if isinstance(result_item, Exception):
                    status_note = f"error_item_processing_failed_in_pipeline: {str(result_item)[:100]}"
                all_classification_results.append({
                    "classification_status": status_note,
                    "raw_output": str(result_item)[:200]
                })
        
        if len(all_classification_results) != num_chunks:
            print(f"  Classification Warning: Output result count ({len(all_classification_results)}) "
                  f"does not match input chunk count ({num_chunks}). Some results may be missing or duplicated.")
            # Pad if results are short (less likely if tqdm consumes the full iterator from pipeline)
            while len(all_classification_results) < num_chunks:
                all_classification_results.append({"classification_status": "error_missing_pipeline_output_item"})
        print("Classification Module: Classification processing completed.")

    except Exception as e_pipeline_call_error:
        print(f"Classification Module Error: A major error occurred during the main pipeline classification call: {e_pipeline_call_error}")
        # If the entire classifier(...) call fails, mark all as errored.
        all_classification_results = [{"classification_status": "error_in_bulk_pipeline_call", "error_message": str(e_pipeline_call_error)}] * num_chunks
            
    return all_classification_results