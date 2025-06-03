# src/pipeline_orchestrator.py
# This module orchestrates the entire data processing pipeline, from document ingestion
# to vector store creation. It coordinates calls to various sub-modules.

import os
import json
import numpy as np
from datetime import datetime
import torch # For checking CUDA availability for device assignment
from vicinity import Metric # For specifying vector store metric
from typing import Dict, Any, Optional # For type hinting

# RAGdoll project-specific imports
from . import ragdoll_utils         # For utility functions like ID sanitization
from . import ragdoll_config        # For default configurations and constants
from . import data_processing_core  # For document preparation, chunking, and classification
from . import embedding_module      # For generating vector embeddings
from . import vector_store_manager  # For managing the Vicinity vector store
from .utils import visualization_utils # For UMAP visualization data generation

def _get_specific_chunker_params(selected_chunker_type: str, global_pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs the specific parameter dictionary for the selected Chonkie chunker.
    It maps CLI/API config keys (from `global_pipeline_config`) to Chonkie-native 
    parameter keys that the chunker constructors in `_init_worker_chunker` expect.
    Defaults are taken from `ragdoll_config.CHUNKER_DEFAULTS`.
    """
    # Start with the 'type', which is essential for _init_worker_chunker
    specific_params_for_worker = {"type": selected_chunker_type}
    # Get the default parameters for the selected chunker type from ragdoll_config
    chunker_defaults_from_config = ragdoll_config.CHUNKER_DEFAULTS.get(selected_chunker_type, {})
    
    # This dictionary will hold the Chonkie-native parameters
    chonkie_native_params: Dict[str, Any] = {} 

    # --- Parameter mapping for each Chonkie chunker type ---
    # The goal is to populate `chonkie_native_params` with keys and values
    # that the actual Chonkie chunker constructors expect.

    if selected_chunker_type == "chonkie_sdpm":
        # Define Chonkie SDPMChunker keys and their corresponding global_config (CLI/API) keys
        param_map = {
            "embedding_model": "chonkie_embedding_model", 
            "chunk_size": "chonkie_target_chunk_size", # Maps RAGdoll's target_chunk_size to Chonkie's chunk_size
            "threshold": "chonkie_similarity_threshold", 
            "min_sentences": "chonkie_min_sentences",
            "mode": "chonkie_mode", 
            "similarity_window": "chonkie_similarity_window",
            "min_chunk_size": "chonkie_min_chunk_size", 
            "skip_window": "chonkie_skip_window",
            # This key is directly from CHUNKER_DEFAULTS, not usually overridden by a specific CLI arg for SDPM
            "min_characters_per_sentence": "min_characters_per_sentence" 
        }
        for chonkie_key, config_key_name in param_map.items():
            # Get value from global_pipeline_config if provided, else from chunker_defaults_from_config
            value = global_pipeline_config.get(config_key_name, chunker_defaults_from_config.get(chonkie_key))
            if value is not None: # Only add if a value is found (either from config or defaults)
                chonkie_native_params[chonkie_key] = value
        # SDPMChunker does not take 'tokenizer_or_token_counter' directly.

    elif selected_chunker_type == "chonkie_semantic":
        param_map = {
            "embedding_model": "chonkie_embedding_model", 
            "chunk_size": "chonkie_target_chunk_size",
            "threshold": "chonkie_similarity_threshold", 
            "min_sentences": "chonkie_min_sentences",
            "mode": "chonkie_mode", 
            "similarity_window": "chonkie_similarity_window",
            "min_chunk_size": "chonkie_min_chunk_size", 
            "min_characters_per_sentence": "min_characters_per_sentence"
        }
        for chonkie_key, config_key_name in param_map.items():
            value = global_pipeline_config.get(config_key_name, chunker_defaults_from_config.get(chonkie_key))
            if value is not None:
                chonkie_native_params[chonkie_key] = value
        # SemanticChunker also does not take 'tokenizer_or_token_counter' directly.

    elif selected_chunker_type == "chonkie_neural":
        param_map = {
            "model": "chonkie_neural_model", 
            "tokenizer": "chonkie_neural_tokenizer", # Chonkie's NeuralChunker might derive this from model
            "min_characters_per_chunk": "chonkie_neural_min_chars", 
            "stride": "chonkie_neural_stride"
        }
        for chonkie_key, config_key_name in param_map.items():
            value = global_pipeline_config.get(config_key_name, chunker_defaults_from_config.get(chonkie_key))
            # Special handling: if NeuralChunker's tokenizer is not set, default to its model name
            if chonkie_key == "tokenizer" and value is None:
                value = global_pipeline_config.get("chonkie_neural_model", chunker_defaults_from_config.get("model"))
            if value is not None:
                chonkie_native_params[chonkie_key] = value

    elif selected_chunker_type == "chonkie_recursive":
        param_map = {
            "tokenizer_or_token_counter": "chonkie_basic_tokenizer",
            "chunk_size": "chonkie_basic_chunk_size",
            "min_characters_per_chunk": "chonkie_basic_min_chars",
            "rules": "chonkie_recursive_rules_recipe_name", # User provides recipe name (e.g., "markdown") via this CLI arg
            "lang": "chonkie_recursive_rules_lang"         # User provides lang for recipe via this CLI arg
        }
        for chonkie_key, config_key_name in param_map.items():
            value = global_pipeline_config.get(config_key_name, chunker_defaults_from_config.get(chonkie_key))
            # 'rules' can be None from defaults if no recipe specified.
            # 'lang' is only relevant if 'rules' is a string (recipe name).
            if value is not None or chonkie_key == "rules": # ensure 'rules' is passed even if None
                chonkie_native_params[chonkie_key] = value
        
        # If 'rules' is not a string (e.g., None or already a RecursiveRules object from elsewhere),
        # then 'lang' is not needed directly for the Chonkie RecursiveChunker constructor
        # as 'lang' is a parameter for `RecursiveRules.from_recipe()`.
        if not isinstance(chonkie_native_params.get("rules"), str):
            chonkie_native_params.pop("lang", None) # Remove 'lang' if 'rules' isn't a recipe string

    elif selected_chunker_type == "chonkie_sentence":
        param_map = {
            "tokenizer_or_token_counter": "chonkie_basic_tokenizer", 
            "chunk_size": "chonkie_basic_chunk_size",
            "chunk_overlap": "chonkie_sentence_overlap", 
            # Map RAGdoll's general 'chonkie_min_sentences' CLI arg if available, else SentenceChunker default
            "min_sentences_per_chunk": "chonkie_min_sentences", 
            # Map RAGdoll's general 'chonkie_basic_min_chars' to SentenceChunker's min_characters_per_sentence
            "min_characters_per_sentence": "chonkie_basic_min_chars" 
        }
        for chonkie_key, config_key_name in param_map.items():
            # For params that might have different CLI arg names based on context (e.g. min_sentences)
            # this attempts to get from the specific CLI arg, then the Chonkie default for that key.
            default_for_key = chunker_defaults_from_config.get(chonkie_key)
            value = global_pipeline_config.get(config_key_name, default_for_key)
            if value is not None:
                chonkie_native_params[chonkie_key] = value

    elif selected_chunker_type == "chonkie_token":
        param_map = {
            "tokenizer": "chonkie_basic_tokenizer", # TokenChunker expects 'tokenizer'
            "chunk_size": "chonkie_basic_chunk_size",
            "chunk_overlap": "chonkie_sentence_overlap" # RAGdoll uses a shared CLI arg for overlap
        }
        for chonkie_key, config_key_name in param_map.items():
            value = global_pipeline_config.get(config_key_name, chunker_defaults_from_config.get(chonkie_key))
            if value is not None:
                chonkie_native_params[chonkie_key] = value
    
    # Add 'return_type': 'texts' to all Chonkie chunkers as RAGdoll expects text output
    chonkie_native_params["return_type"] = "texts"
    
    specific_params_for_worker.update(chonkie_native_params)
    return specific_params_for_worker

def run_full_processing_pipeline(config: Dict[str, Any]) -> bool:
    """
    Executes the full data processing pipeline based on the provided configuration.
    Steps include: printing config, text extraction, chunking, classification (optional),
    embedding generation, vector store creation, and visualization data preparation (optional).
    """
    print_pipeline_config(config) 
    
    docs_folder_path = config.get("docs_folder", ragdoll_config.DEFAULT_SOURCE_DOCS_DIR)
    if not os.path.exists(docs_folder_path) or not os.listdir(docs_folder_path):
        print(f"Pipeline Orchestrator Error: Documents folder '{docs_folder_path}' is empty or non-existent. Aborting.")
        return False
    
    vector_data_dir_path = config["vector_data_dir"]
    if not os.path.exists(vector_data_dir_path): 
        print(f"Pipeline Orchestrator: Output directory '{vector_data_dir_path}' does not exist. Creating it.")
        os.makedirs(vector_data_dir_path, exist_ok=True)

    # Prepare the specific parameter dictionary for the chosen chunker
    selected_chunker_type = config["chunker_type"]
    specific_chunker_params_for_core = _get_specific_chunker_params(selected_chunker_type, config)

    print("\nORCHESTRATOR: Starting Document Preparation, Chunking, and Classification...")
    # Call the core data processing function
    text_chunks, chunk_ids, detailed_metadata = \
        data_processing_core.prepare_documents_and_chunks(
            docs_input_folder=docs_folder_path, 
            chunker_config=specific_chunker_params_for_core, # Pass the tailored chunker config
            num_workers_cfg=config["num_processing_workers"], 
            enable_clf_cfg=config["enable_classification"], 
            clf_model_name_cfg=config.get("classifier_model_name"), # Optional, might be None
            clf_labels_cfg=config.get("candidate_labels"),         # Optional, might be None
            clf_batch_size_cfg=config.get("classification_batch_size", ragdoll_config.DEFAULT_CLASSIFICATION_BATCH_SIZE), 
            gpu_id_cfg=config["gpu_device_id"], 
            verbose_logging_cfg=config.get("verbose", False)
        )
    
    if not text_chunks: 
        print("Pipeline Orchestrator: Pipeline aborted - No text chunks were generated after processing documents.")
        return False
    print(f"Pipeline Orchestrator: Text processing & chunking complete. Total chunks generated: {len(text_chunks)}")

    # --- Embedding Generation ---
    print("\nORCHESTRATOR: Starting Embedding Generation...")
    main_embedding_model_name = config["embedding_model_name"]
    
    # Infer parameters: (type, dimension, metric_enum, class_type)
    # Ensure variable names here match what's used below.
    inferred_type_str, inferred_dim_val, inferred_metric_enum_val, inferred_class_str = \
        embedding_module.infer_embedding_model_params(main_embedding_model_name)
    
    # User overrides from config
    user_model_type_override = config.get("embedding_model_type") 
    user_dimension_override = config.get("default_embedding_dim") # Can be None
    
    # Determine device for embedding generation
    pipeline_embedding_device = "cpu" 
    if config["gpu_device_id"] >= 0:
        if torch.cuda.is_available(): 
            pipeline_embedding_device = f"cuda:{config['gpu_device_id']}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): 
            pipeline_embedding_device = "mps"
        else: 
            print(f"Pipeline Orchestrator Warning: GPU {config['gpu_device_id']} for embeddings not available, using CPU.")

    # Generate embeddings for all text chunks
    chunk_vectors, actual_embedding_dimension = embedding_module.generate_embeddings(
        text_list=text_chunks, 
        model_name_or_path=main_embedding_model_name, 
        model_type_override=user_model_type_override,         # Pass user's explicit type override
        expected_dimension_override=user_dimension_override,  # Pass user's explicit dim override
        # Pass all inferred parameters:
        inferred_model_type=inferred_type_str,      
        inferred_model_class=inferred_class_str,      
        inferred_dimension=inferred_dim_val,     # Corrected variable name passed here    
        device=pipeline_embedding_device
    )
    
    if chunk_vectors.shape[0] == 0: 
        print("Pipeline Orchestrator: Pipeline aborted - No embeddings were generated.")
        return False
    print(f"Pipeline Orchestrator: Embeddings generated successfully. Actual dimension: {actual_embedding_dimension}")

    # --- Vector Store Creation ---
    print("\nORCHESTRATOR: Initializing and Saving Vector Store...")
    vsm = vector_store_manager.VectorStoreManager(
        vector_data_dir=config["vector_data_dir"], 
        vicinity_store_name=ragdoll_config.VECTOR_STORE_SUBDIR_NAME
    )
    
    final_vicinity_metric_to_use = inferred_metric_enum_val # Start with inferred metric
    user_specified_metric_str = config.get("vicinity_metric")
    if user_specified_metric_str:
        try: 
            final_vicinity_metric_to_use = getattr(Metric, user_specified_metric_str.upper())
        except AttributeError: 
            print(f"Pipeline Orchestrator Warning: Invalid Vicinity metric '{user_specified_metric_str}'. Using inferred: {inferred_metric_enum_val}.")
    
    # Prepare metadata for the vector store
    specific_chunker_params_for_core = _get_specific_chunker_params(config["chunker_type"], config) # Already called earlier
    store_chunk_params_metadata = specific_chunker_params_for_core.copy()
    if 'tokenizer_or_token_counter' in store_chunk_params_metadata and \
       not isinstance(store_chunk_params_metadata['tokenizer_or_token_counter'], str):
        store_chunk_params_metadata['tokenizer_or_token_counter'] = str(type(store_chunk_params_metadata['tokenizer_or_token_counter']))


    store_processing_metadata = {
        "embedding_model_name": main_embedding_model_name, 
        # Use the type that was actually used for loading the model:
        "embedding_model_type": user_model_type_override or inferred_type_str, 
        "embedding_model_class": inferred_class_str, 
        "embedding_vector_dimension": actual_embedding_dimension, # Use the dimension of generated vectors
        "vicinity_metric": str(final_vicinity_metric_to_use).split('.')[-1],
        "source_docs_folder": config["docs_folder"], 
        "pipeline_run_date": datetime.now().isoformat(),
        "classification_enabled": config["enable_classification"],
        "classifier_model_name": config.get("classifier_model_name") if config["enable_classification"] else None,
        "chunker_type": config["chunker_type"], 
        "chunk_params": store_chunk_params_metadata, 
        "pipeline_config_summary": {k:v for k,v in config.items() if k not in ["candidate_labels", "config_file"]}, 
    }
    # ... (rest of metadata population and store creation as before) ...
    if config["enable_classification"] and config.get("candidate_labels"):
        labels_list = config["candidate_labels"]
        store_processing_metadata["pipeline_config_summary"]["candidate_labels_count"] = len(labels_list)
        store_processing_metadata["pipeline_config_summary"]["candidate_labels_preview"] = \
            labels_list[:10] + ["..."] if len(labels_list) > 10 else labels_list

    store_created_successfully = vsm.create_and_save_store(
        chunk_vectors, chunk_ids, final_vicinity_metric_to_use, 
        store_processing_metadata, config.get("overwrite", False)
    )
    if not store_created_successfully: 
        print("Pipeline Orchestrator: Pipeline aborted - Vector store creation failed.")
        return False
    
    save_orchestrator_supporting_data(config["vector_data_dir"], text_chunks, chunk_ids, detailed_metadata, chunk_vectors)
    
    if config.get("prepare_viz_data", False) and chunk_vectors.shape[0] > 0:
        print("\nORCHESTRATOR: Preparing Visualization Data using UMAP...")
        visualization_utils.save_data_for_visualization(
            vector_data_dir=config["vector_data_dir"], 
            chunk_vectors=chunk_vectors, 
            metadata_list=detailed_metadata, 
            chunk_texts_list=text_chunks, 
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
    """
    Saves supporting data (text chunks, IDs, detailed metadata, and raw vectors)
    as JSON and NumPy files in the specified vector data directory.
    """
    print(f"\nPipeline Orchestrator: Saving supporting data files to '{vector_data_dir_path}'...")
    try:
        # Save text chunks to a JSON file
        with open(os.path.join(vector_data_dir_path, ragdoll_config.TEXT_CHUNKS_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(all_text_chunks, f, indent=2) # Use indent for readability
        
        # Save chunk IDs to a JSON file
        with open(os.path.join(vector_data_dir_path, ragdoll_config.CHUNK_IDS_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(all_chunk_ids, f, indent=2)
            
        # Save detailed metadata for each chunk to a JSON file
        with open(os.path.join(vector_data_dir_path, ragdoll_config.DETAILED_CHUNK_METADATA_FILENAME), 'w', encoding='utf-8') as f:
            json.dump(all_detailed_metadata, f, indent=2)
            
        # Save the raw chunk vectors as a NumPy .npy file if any vectors were generated
        if all_chunk_vectors.shape[0] > 0:
            np.save(os.path.join(vector_data_dir_path, ragdoll_config.CHUNK_VECTORS_FILENAME), all_chunk_vectors)
            
        print("Pipeline Orchestrator: Supporting data files saved successfully.")
    except Exception as e:
        print(f"Pipeline Orchestrator Error: Failed to save one or more supporting data files: {e}")

def print_pipeline_config(config: Dict[str, Any]):
    """Prints a summary of the pipeline configuration being used."""
    print(f"--- Pipeline Orchestrator Starting with Configuration ---")
    print(f"  Documents Input Folder: {config.get('docs_folder', ragdoll_config.DEFAULT_SOURCE_DOCS_DIR)}")
    print(f"  Vector Data Output Dir: {config.get('vector_data_dir', 'vector_store_data')}") # Default if not in config
    print(f"  Overwrite Existing Data: {config.get('overwrite', False)}")
    print(f"  GPU Device ID for Processing: {config.get('gpu_device_id', ragdoll_config.DEFAULT_GPU_DEVICE_ID_PROCESSING)}")
    print(f"  Verbose Logging: {config.get('verbose',False)}")
    print(f"  Pipeline Embedding Model: {config.get('embedding_model_name', ragdoll_config.DEFAULT_PIPELINE_EMBEDDING_MODEL)}")
    
    chunker_type_selected = config.get('chunker_type', ragdoll_config.DEFAULT_CHUNKER_TYPE)
    print(f"  Chunker Type Selected: {chunker_type_selected.upper()}")
    print(f"  Num Processing Workers for Chunking: {config.get('num_processing_workers', ragdoll_config.DEFAULT_CHUNK_PROCESSING_WORKERS)}")
    
    # Get default parameters for the selected chunker to print them
    default_params_for_selected_chunker = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_selected, {})
    
    # Print specific parameters for the chosen chunker type
    if chunker_type_selected == "chonkie_sdpm":
        print(f"    Chonkie (SDPM) Internal Embedding Model: {config.get('chonkie_embedding_model', default_params_for_selected_chunker.get('embedding_model'))}")
        print(f"    Chonkie (SDPM) Target Chunk Size (tokens): {config.get('chonkie_target_chunk_size', default_params_for_selected_chunker.get('chunk_size'))}")
        print(f"    Chonkie (SDPM) Similarity Threshold: {config.get('chonkie_similarity_threshold', default_params_for_selected_chunker.get('threshold'))}")
        print(f"    Chonkie (SDPM) Min Sentences per Chunk: {config.get('chonkie_min_sentences', default_params_for_selected_chunker.get('min_sentences'))}")
        print(f"    Chonkie (SDPM) Processing Mode: {config.get('chonkie_mode', default_params_for_selected_chunker.get('mode'))}")
        print(f"    Chonkie (SDPM) Similarity Window: {config.get('chonkie_similarity_window', default_params_for_selected_chunker.get('similarity_window'))}")
        print(f"    Chonkie (SDPM) Min Final Chunk Size (tokens): {config.get('chonkie_min_chunk_size', default_params_for_selected_chunker.get('min_chunk_size'))}")
        print(f"    Chonkie (SDPM) Skip Window for Peak Finding: {config.get('chonkie_skip_window', default_params_for_selected_chunker.get('skip_window'))}")
    elif chunker_type_selected == "chonkie_semantic":
        print(f"    Chonkie (Semantic) Internal Embedding Model: {config.get('chonkie_embedding_model', default_params_for_selected_chunker.get('embedding_model'))}")
        print(f"    Chonkie (Semantic) Target Chunk Size (tokens): {config.get('chonkie_target_chunk_size', default_params_for_selected_chunker.get('chunk_size'))}")
        # ... (add other semantic params similarly) ...
    elif chunker_type_selected == "chonkie_neural":
        print(f"    Chonkie (Neural) Segmentation Model: {config.get('chonkie_neural_model', default_params_for_selected_chunker.get('model'))}")
        # Default tokenizer for Neural to be same as model if not specified
        neural_tokenizer = config.get('chonkie_neural_tokenizer', config.get('chonkie_neural_model', default_params_for_selected_chunker.get('tokenizer')))
        print(f"    Chonkie (Neural) Tokenizer (for model): {neural_tokenizer}")
        print(f"    Chonkie (Neural) Inference Stride: {config.get('chonkie_neural_stride', default_params_for_selected_chunker.get('stride'))}")
        print(f"    Chonkie (Neural) Min Characters per Chunk: {config.get('chonkie_neural_min_chars', default_params_for_selected_chunker.get('min_characters_per_chunk'))}")
    elif chunker_type_selected == "chonkie_recursive":
        print(f"    Chonkie (Recursive) Tokenizer/Counter: {config.get('chonkie_basic_tokenizer', default_params_for_selected_chunker.get('tokenizer_or_token_counter'))}")
        print(f"    Chonkie (Recursive) Target Chunk Size (tokens): {config.get('chonkie_basic_chunk_size', default_params_for_selected_chunker.get('chunk_size'))}")
        print(f"    Chonkie (Recursive) Min Characters per Chunk: {config.get('chonkie_basic_min_chars', default_params_for_selected_chunker.get('min_characters_per_chunk'))}")
        # Print recipe name if specified, otherwise indicate default rules
        rules_recipe_name = config.get('chonkie_recursive_rules_recipe_name', default_params_for_selected_chunker.get('rules'))
        if isinstance(rules_recipe_name, str): # If a recipe name (like "markdown") is used
            print(f"    Chonkie (Recursive) Rules Recipe: {rules_recipe_name}")
            print(f"    Chonkie (Recursive) Rules Language: {config.get('chonkie_recursive_rules_lang', 'en')}") # Default lang to 'en' if recipe used
        else: # If rules is None or an actual RecursiveRules object, just indicate default behavior
             print(f"    Chonkie (Recursive) Rules: Using Chonkie's default recursive rules.")
    elif chunker_type_selected == "chonkie_sentence":
        print(f"    Chonkie (Sentence) Tokenizer/Counter: {config.get('chonkie_basic_tokenizer', default_params_for_selected_chunker.get('tokenizer_or_token_counter'))}")
        print(f"    Chonkie (Sentence) Target Chunk Size (Tokens): {config.get('chonkie_basic_chunk_size', default_params_for_selected_chunker.get('chunk_size'))}")
        print(f"    Chonkie (Sentence) Chunk Overlap: {config.get('chonkie_sentence_overlap', default_params_for_selected_chunker.get('chunk_overlap'))}")
        print(f"    Chonkie (Sentence) Min Sentences per Chunk: {config.get('chonkie_min_sentences', default_params_for_selected_chunker.get('min_sentences_per_chunk'))}")
        print(f"    Chonkie (Sentence) Min Characters per Sentence: {config.get('chonkie_basic_min_chars', default_params_for_selected_chunker.get('min_characters_per_sentence'))}")
    elif chunker_type_selected == "chonkie_token":
        print(f"    Chonkie (Token) Tokenizer: {config.get('chonkie_basic_tokenizer', default_params_for_selected_chunker.get('tokenizer'))}")
        print(f"    Chonkie (Token) Target Chunk Size (Tokens): {config.get('chonkie_basic_chunk_size', default_params_for_selected_chunker.get('chunk_size'))}")
        print(f"    Chonkie (Token) Chunk Overlap: {config.get('chonkie_sentence_overlap', default_params_for_selected_chunker.get('chunk_overlap'))}")

    classification_enabled = config.get('enable_classification', False)
    print(f"  Classification Enabled: {classification_enabled}", end="")
    if classification_enabled: 
        print(f" (Classifier Model: {config.get('classifier_model_name', ragdoll_config.DEFAULT_CLASSIFIER_MODEL)})")
    else: 
        print() # Newline if classification is not enabled
        
    visualization_enabled = config.get('prepare_viz_data', False)
    print(f"  Prepare Visualization Data: {visualization_enabled}")
    if visualization_enabled:
        print(f"    Visualization Output File: {config.get('viz_output_file', ragdoll_config.VISUALIZATION_DATA_FILENAME)}")
        print(f"    UMAP Neighbors: {config.get('umap_neighbors', ragdoll_config.DEFAULT_UMAP_NEIGHBORS)}")
        print(f"    UMAP Min Distance: {config.get('umap_min_dist', ragdoll_config.DEFAULT_UMAP_MIN_DIST)}")
    print("-" * 50)