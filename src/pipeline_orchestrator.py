# src/pipeline_orchestrator.py
# This module orchestrates the full data processing pipeline, from document ingestion
# to vector store creation. It coordinates calls to various processing modules.

import os
import json
import numpy as np
from datetime import datetime
import torch # For checking GPU availability and setting device for embeddings
from vicinity import Metric # For specifying the distance metric in Vicinity vector store
from typing import Dict, Any # For type hinting

# Project-specific imports for utilities, configurations, and core processing modules
# Assuming these are in the same 'src' directory or 'src' is in PYTHONPATH
# If 'core_logic' and 'utils' are subdirectories of 'src', adjust paths like:
# from .core_logic import data_processing_core 
# from .utils import visualization_utils
from . import ragdoll_utils
from . import ragdoll_config
from . import data_processing_core # Refactored version
from . import embedding_module
from . import vector_store_manager
from .utils import visualization_utils # New location for UMAP generation


def _get_specific_chunker_params(selected_chunker_type: str, global_pipeline_config: dict) -> Dict[str, Any]:
    """
    Constructs the specific parameter dictionary for the selected Chonkie chunker.
    It maps CLI/API configuration keys (from global_pipeline_config) to the 
    Chonkie-native parameter keys expected by the chunker constructors in data_processing_core.
    It also incorporates default values from ragdoll_config.CHUNKER_DEFAULTS.
    """
    specific_params_for_worker = {"type": selected_chunker_type} 
    chunker_defaults_from_config = ragdoll_config.CHUNKER_DEFAULTS.get(selected_chunker_type, {})
    chonkie_native_params: Dict[str, Any] = {} 

    if selected_chunker_type == "chonkie_sdpm":
        chonkie_native_params = {
            "embedding_model": global_pipeline_config.get("chonkie_embedding_model", chunker_defaults_from_config.get("embedding_model")),
            "chunk_size": global_pipeline_config.get("chonkie_target_chunk_size", chunker_defaults_from_config.get("chunk_size")),
            "threshold": global_pipeline_config.get("chonkie_similarity_threshold", chunker_defaults_from_config.get("threshold")),
            "min_sentences": global_pipeline_config.get("chonkie_min_sentences", chunker_defaults_from_config.get("min_sentences")),
            "mode": global_pipeline_config.get("chonkie_mode", chunker_defaults_from_config.get("mode")),
            "similarity_window": global_pipeline_config.get("chonkie_similarity_window", chunker_defaults_from_config.get("similarity_window")),
            "min_chunk_size": global_pipeline_config.get("chonkie_min_chunk_size", chunker_defaults_from_config.get("min_chunk_size")),
            "skip_window": global_pipeline_config.get("chonkie_skip_window", chunker_defaults_from_config.get("skip_window")),
            "min_characters_per_sentence": chunker_defaults_from_config.get("min_characters_per_sentence"),
            "tokenizer_or_token_counter": chunker_defaults_from_config.get("tokenizer_or_token_counter"),
        }
    elif selected_chunker_type == "chonkie_semantic":
        chonkie_native_params = {
            "embedding_model": global_pipeline_config.get("chonkie_embedding_model", chunker_defaults_from_config.get("embedding_model")),
            "chunk_size": global_pipeline_config.get("chonkie_target_chunk_size", chunker_defaults_from_config.get("chunk_size")),
            "threshold": global_pipeline_config.get("chonkie_similarity_threshold", chunker_defaults_from_config.get("threshold")),
            "min_sentences": global_pipeline_config.get("chonkie_min_sentences", chunker_defaults_from_config.get("min_sentences")),
            "mode": global_pipeline_config.get("chonkie_mode", chunker_defaults_from_config.get("mode")),
            "similarity_window": global_pipeline_config.get("chonkie_similarity_window", chunker_defaults_from_config.get("similarity_window")),
            "min_chunk_size": global_pipeline_config.get("chonkie_min_chunk_size", chunker_defaults_from_config.get("min_chunk_size")),
            "min_characters_per_sentence": chunker_defaults_from_config.get("min_characters_per_sentence"),
            "tokenizer_or_token_counter": chunker_defaults_from_config.get("tokenizer_or_token_counter"),
        }
    elif selected_chunker_type == "chonkie_neural":
        chonkie_native_params = {
            "model": global_pipeline_config.get("chonkie_neural_model", chunker_defaults_from_config.get("model")),
            "tokenizer": global_pipeline_config.get("chonkie_neural_tokenizer", 
                                         global_pipeline_config.get("chonkie_neural_model", chunker_defaults_from_config.get("tokenizer"))),
            "min_characters_per_chunk": global_pipeline_config.get("chonkie_neural_min_chars", chunker_defaults_from_config.get("min_characters_per_chunk")),
            "stride": global_pipeline_config.get("chonkie_neural_stride", chunker_defaults_from_config.get("stride")),
        }
    elif selected_chunker_type == "chonkie_recursive":
        chonkie_native_params = {
            "tokenizer_or_token_counter": global_pipeline_config.get("chonkie_basic_tokenizer", chunker_defaults_from_config.get("tokenizer_or_token_counter")),
            "chunk_size": global_pipeline_config.get("chonkie_basic_chunk_size", chunker_defaults_from_config.get("chunk_size")),
            "min_characters_per_chunk": global_pipeline_config.get("chonkie_basic_min_chars", chunker_defaults_from_config.get("min_characters_per_chunk")),
        }
        if "chonkie_recursive_rules_recipe" in global_pipeline_config: # e.g., a new CLI arg --chonkie-recursive-rules-recipe markdown
            chonkie_native_params["rules"] = global_pipeline_config["chonkie_recursive_rules_recipe"] # Pass "markdown" as string
            if "chonkie_recursive_lang" in global_pipeline_config: # e.g. --chonkie-recursive-lang en
                chonkie_native_params["lang"] = global_pipeline_config["chonkie_recursive_lang"]
    elif selected_chunker_type == "chonkie_sentence":
        chonkie_native_params = {
            "tokenizer_or_token_counter": global_pipeline_config.get("chonkie_basic_tokenizer", chunker_defaults_from_config.get("tokenizer_or_token_counter")),
            "chunk_size": global_pipeline_config.get("chonkie_basic_chunk_size", chunker_defaults_from_config.get("chunk_size")),
            "chunk_overlap": global_pipeline_config.get("chonkie_sentence_overlap", chunker_defaults_from_config.get("chunk_overlap")),
            "min_sentences_per_chunk": global_pipeline_config.get("chonkie_min_sentences", chunker_defaults_from_config.get("min_sentences_per_chunk")),
            "min_characters_per_sentence": global_pipeline_config.get("chonkie_basic_min_chars", chunker_defaults_from_config.get("min_characters_per_sentence")),
        }
    elif selected_chunker_type == "chonkie_token":
        chonkie_native_params = {
            "tokenizer": global_pipeline_config.get("chonkie_basic_tokenizer", chunker_defaults_from_config.get("tokenizer")),
            "chunk_size": global_pipeline_config.get("chonkie_basic_chunk_size", chunker_defaults_from_config.get("chunk_size")),
            "chunk_overlap": global_pipeline_config.get("chonkie_sentence_overlap", chunker_defaults_from_config.get("chunk_overlap")),
        }
    
    for key, value in chonkie_native_params.items():
        specific_params_for_worker[key] = value
        
    return specific_params_for_worker

def run_full_processing_pipeline(config: dict) -> bool:
    """
    Executes the entire data processing pipeline.
    """
    print_pipeline_config(config) 
    
    if not os.path.exists(config["docs_folder"]) or not os.listdir(config["docs_folder"]):
        print(f"Pipeline Orchestrator Error: Docs folder '{config['docs_folder']}' empty or non-existent."); return False
    if not os.path.exists(config["vector_data_dir"]): 
        os.makedirs(config["vector_data_dir"], exist_ok=True)

    selected_chunker_type = config["chunker_type"]
    specific_chunker_params_for_core = _get_specific_chunker_params(selected_chunker_type, config)

    # --- Step 1: Document Preparation, Chunking, and Optional Classification ---
    # This step is now primarily handled by data_processing_core, which internally calls
    # file_parser for extraction and classification_module for classification.
    print("\nORCHESTRATOR: Starting Document Preparation, Chunking, and Classification...")
    text_chunks, chunk_ids, detailed_metadata = \
        data_processing_core.prepare_documents_and_chunks(
            docs_input_folder=config["docs_folder"], 
            chunker_config=specific_chunker_params_for_core,
            num_workers_cfg=config["num_processing_workers"], 
            enable_clf_cfg=config["enable_classification"], 
            clf_model_name_cfg=config.get("classifier_model_name"),
            clf_labels_cfg=config.get("candidate_labels"), 
            clf_batch_size_cfg=config["classification_batch_size"], 
            gpu_id_cfg=config["gpu_device_id"], 
            verbose_logging_cfg=config.get("verbose", False)
        )
    if not text_chunks: 
        print("Pipeline Orchestrator: Pipeline aborted - No text chunks generated."); return False
    print(f"Pipeline Orchestrator: Text processing & chunking complete. Chunks: {len(text_chunks)}")

    # --- Step 2: Embedding Generation ---
    print("\nORCHESTRATOR: Starting Embedding Generation...")
    main_emb_model_name = config["embedding_model_name"]
    inferred_model_type, inferred_dimension, inferred_metric_enum, inferred_model_class = \
        embedding_module.infer_embedding_model_params(main_emb_model_name)
    
    final_model_type = config.get("embedding_model_type") or inferred_model_type
    final_default_dimension = config.get("default_embedding_dim") if config.get("default_embedding_dim") is not None else inferred_dimension
    
    pipeline_embedding_device = "cpu"
    if config["gpu_device_id"] >= 0:
        if torch.cuda.is_available(): 
            pipeline_embedding_device = f"cuda:{config['gpu_device_id']}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
            pipeline_embedding_device = "mps"
        else:
             print(f"Pipeline Orchestrator: GPU {config['gpu_device_id']} not available for embeddings, using CPU.")


    chunk_vectors, actual_embedding_dimension = embedding_module.generate_embeddings(
        text_chunks, main_emb_model_name, final_model_type, final_default_dimension, 
        inferred_model_class, device=pipeline_embedding_device
    )
    if chunk_vectors.shape[0] == 0: 
        print("Pipeline Orchestrator: Pipeline aborted - No embeddings generated."); return False
    print(f"Pipeline Orchestrator: Embeddings generated. Dimension: {actual_embedding_dimension}")

    # --- Step 3: Vector Store Creation ---
    print("\nORCHESTRATOR: Initializing and Saving Vector Store...")
    vsm = vector_store_manager.VectorStoreManager(config["vector_data_dir"], ragdoll_config.VECTOR_STORE_SUBDIR_NAME)
    
    final_vicinity_metric = inferred_metric_enum
    if config.get("vicinity_metric"):
        try: 
            final_vicinity_metric = getattr(Metric, config["vicinity_metric"].upper())
        except AttributeError: 
            print(f"Pipeline Orchestrator Warning: Invalid Vicinity metric '{config['vicinity_metric']}'. Using inferred: {inferred_metric_enum}.")
    
    store_chunk_params_for_metadata = specific_chunker_params_for_core.copy() 

    store_processing_metadata = {
        "embedding_model_name": main_emb_model_name, 
        "embedding_model_type": final_model_type,
        "embedding_model_class": inferred_model_class, 
        "embedding_vector_dimension": actual_embedding_dimension,
        "vicinity_metric": str(final_vicinity_metric).split('.')[-1],
        "source_docs_folder": config["docs_folder"], 
        "pipeline_run_date": datetime.now().isoformat(),
        "classification_enabled": config["enable_classification"],
        "classifier_model_name": config.get("classifier_model_name") if config["enable_classification"] else None,
        "chunker_type": selected_chunker_type, 
        "chunk_params": store_chunk_params_for_metadata, 
        "pipeline_config_summary": {k:v for k,v in config.items() if k not in ["candidate_labels", "config_file"]}, 
    }
    if config["enable_classification"] and config.get("candidate_labels"):
        labels_list = config["candidate_labels"]
        store_processing_metadata["pipeline_config_summary"]["candidate_labels_count"] = len(labels_list)
        store_processing_metadata["pipeline_config_summary"]["candidate_labels_preview"] = \
            labels_list[:10] + ["..."] if len(labels_list) > 10 else labels_list

    store_created_successfully = vsm.create_and_save_store(
        chunk_vectors, chunk_ids, final_vicinity_metric, 
        store_processing_metadata, config.get("overwrite", False)
    )
    if not store_created_successfully: 
        print("Pipeline Orchestrator: Pipeline aborted - Vector store creation failed."); return False
    
    # --- Step 4: Save Supporting Data Files ---
    save_orchestrator_supporting_data(config["vector_data_dir"], text_chunks, chunk_ids, detailed_metadata, chunk_vectors)
    
    # --- Step 5: Optional UMAP Visualization Data Generation ---
    # This now calls the dedicated utility in visualization_utils.
    if config.get("prepare_viz_data", False) and chunk_vectors.shape[0] > 0:
        print("\nORCHESTRATOR: Preparing Visualization Data using UMAP...")
        visualization_utils.save_data_for_visualization( # Call the moved function
            vector_data_dir=config["vector_data_dir"], 
            chunk_vectors=chunk_vectors, 
            metadata_list=detailed_metadata, 
            chunk_texts_list=text_chunks, 
            # Defaults for viz params are now handled by save_data_for_visualization itself using ragdoll_config
            output_viz_filename=config.get("viz_output_file", ragdoll_config.VISUALIZATION_DATA_FILENAME), 
            umap_n_neighbors=config.get("umap_neighbors", ragdoll_config.DEFAULT_UMAP_NEIGHBORS), 
            umap_min_dist_val=config.get("umap_min_dist", ragdoll_config.DEFAULT_UMAP_MIN_DIST), 
            umap_metric_val=config.get("umap_metric", ragdoll_config.DEFAULT_UMAP_METRIC)
        )
        
    print("\n--- Pipeline Orchestrator Finished Successfully ---")
    return True

def save_orchestrator_supporting_data(vector_data_dir_path: str, 
                                      all_text_chunks: list, 
                                      all_chunk_ids: list, 
                                      all_detailed_metadata: list, 
                                      all_chunk_vectors: np.ndarray):
    """Saves processed data (chunks, IDs, metadata, vectors) to disk."""
    print(f"\nPipeline Orchestrator: Saving supporting data to '{vector_data_dir_path}'...")
    try:
        with open(os.path.join(vector_data_dir_path, ragdoll_config.TEXT_CHUNKS_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(all_text_chunks, f, indent=2)
        with open(os.path.join(vector_data_dir_path, ragdoll_config.CHUNK_IDS_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(all_chunk_ids, f, indent=2)
        with open(os.path.join(vector_data_dir_path, ragdoll_config.DETAILED_CHUNK_METADATA_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(all_detailed_metadata, f, indent=2)
        if all_chunk_vectors.shape[0] > 0:
            np.save(os.path.join(vector_data_dir_path, ragdoll_config.CHUNK_VECTORS_FILENAME), all_chunk_vectors)
        print("Pipeline Orchestrator: Supporting data files saved.")
    except Exception as e:
        print(f"Pipeline Orchestrator Error: Failed to save supporting data: {e}")

def print_pipeline_config(config: dict):
    """Prints a summary of the pipeline configuration."""
    print(f"--- Pipeline Orchestrator Starting with Configuration ---")
    print(f"  Documents Input Folder: {config.get('docs_folder')}")
    print(f"  Vector Data Output Dir: {config.get('vector_data_dir')}")
    print(f"  Overwrite Existing Data: {config.get('overwrite', False)}")
    print(f"  GPU Device ID for Processing: {config.get('gpu_device_id')}") # This is now ragdoll_config.DEFAULT_GPU_DEVICE_ID_PROCESSING
    print(f"  Verbose Logging: {config.get('verbose',False)}")
    print(f"  Pipeline Embedding Model: {config.get('embedding_model_name')}")
    
    chunker_type = config.get('chunker_type')
    print(f"  Chunker Type Selected: {chunker_type.upper() if chunker_type else 'N/A'}")
    print(f"  Num Processing Workers for Chunking: {config.get('num_processing_workers')}")
    
    # Display specific parameters for the selected chunker type from the main 'config' dict
    # These names match CLI/API arguments.
    if chunker_type == "chonkie_sdpm":
        print(f"    Chonkie (SDPM) Internal Embedding Model: {config.get('chonkie_embedding_model')}")
        print(f"    Chonkie (SDPM) Target Chunk Size (tokens): {config.get('chonkie_target_chunk_size')}")
        print(f"    Chonkie (SDPM) Similarity Threshold: {config.get('chonkie_similarity_threshold')}")
        print(f"    Chonkie (SDPM) Min Sentences per Chunk: {config.get('chonkie_min_sentences')}")
        print(f"    Chonkie (SDPM) Processing Mode: {config.get('chonkie_mode')}")
        print(f"    Chonkie (SDPM) Similarity Window: {config.get('chonkie_similarity_window')}")
        print(f"    Chonkie (SDPM) Min Final Chunk Size (tokens): {config.get('chonkie_min_chunk_size')}")
        print(f"    Chonkie (SDPM) Skip Window for Peak Finding: {config.get('chonkie_skip_window')}")
    elif chunker_type == "chonkie_semantic":
        print(f"    Chonkie (Semantic) Internal Embedding Model: {config.get('chonkie_embedding_model')}")
        print(f"    Chonkie (Semantic) Target Chunk Size (tokens): {config.get('chonkie_target_chunk_size')}")
        print(f"    Chonkie (Semantic) Similarity Threshold: {config.get('chonkie_similarity_threshold')}")
        print(f"    Chonkie (Semantic) Min Sentences per Chunk: {config.get('chonkie_min_sentences')}")
        print(f"    Chonkie (Semantic) Processing Mode: {config.get('chonkie_mode')}")
        print(f"    Chonkie (Semantic) Similarity Window: {config.get('chonkie_similarity_window')}")
        print(f"    Chonkie (Semantic) Min Final Chunk Size (tokens): {config.get('chonkie_min_chunk_size')}")
    elif chunker_type == "chonkie_neural":
        print(f"    Chonkie (Neural) Segmentation Model: {config.get('chonkie_neural_model')}")
        print(f"    Chonkie (Neural) Tokenizer (for model): {config.get('chonkie_neural_tokenizer', config.get('chonkie_neural_model'))}")
        print(f"    Chonkie (Neural) Inference Stride: {config.get('chonkie_neural_stride')}")
        print(f"    Chonkie (Neural) Min Characters per Chunk: {config.get('chonkie_neural_min_chars')}")
    elif chunker_type == "chonkie_recursive":
        print(f"    Chonkie (Recursive) Tokenizer/Counter: {config.get('chonkie_basic_tokenizer')}")
        print(f"    Chonkie (Recursive) Target Chunk Size (tokens): {config.get('chonkie_basic_chunk_size')}")
        print(f"    Chonkie (Recursive) Min Characters per Chunk: {config.get('chonkie_basic_min_chars')}")
        # Default rules from ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["rules"] will be used if not overridden.
        print(f"    Chonkie (Recursive) Rules: {ragdoll_config.CHUNKER_DEFAULTS['chonkie_recursive'].get('rules', 'Chonkie Default')}")
    elif chunker_type == "chonkie_sentence":
        print(f"    Chonkie (Sentence) Tokenizer/Counter: {config.get('chonkie_basic_tokenizer')}")
        print(f"    Chonkie (Sentence) Target Chunk Size (tokens): {config.get('chonkie_basic_chunk_size')}")
        print(f"    Chonkie (Sentence) Chunk Overlap: {config.get('chonkie_sentence_overlap')}")
        print(f"    Chonkie (Sentence) Min Sentences per Chunk: {config.get('chonkie_min_sentences')}") # Shared CLI arg
        print(f"    Chonkie (Sentence) Min Characters per Sentence: {config.get('chonkie_basic_min_chars')}")
    elif chunker_type == "chonkie_token":
        print(f"    Chonkie (Token) Tokenizer: {config.get('chonkie_basic_tokenizer')}")
        print(f"    Chonkie (Token) Target Chunk Size (tokens): {config.get('chonkie_basic_chunk_size')}")
        print(f"    Chonkie (Token) Chunk Overlap: {config.get('chonkie_sentence_overlap')}") # Shared CLI arg

    print(f"  Classification Enabled: {config.get('enable_classification')}", end="")
    if config.get('enable_classification'): 
        print(f" (Classifier Model: {config.get('classifier_model_name')})")
    else: 
        print() 
        
    print(f"  Prepare Visualization Data: {config.get('prepare_viz_data')}")
    if config.get('prepare_viz_data'):
        print(f"    Visualization Output File: {config.get('viz_output_file', ragdoll_config.VISUALIZATION_DATA_FILENAME)}")
        print(f"    UMAP Neighbors: {config.get('umap_neighbors', ragdoll_config.DEFAULT_UMAP_NEIGHBORS)}")
        print(f"    UMAP Min Distance: {config.get('umap_min_dist', ragdoll_config.DEFAULT_UMAP_MIN_DIST)}")
    print("-" * 50)