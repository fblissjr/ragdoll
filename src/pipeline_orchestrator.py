# pipeline_orchestrator.py
import os
import json
import numpy as np
from datetime import datetime
import torch
from vicinity import Metric 

import common_utils
import data_processing_core
import embedding_module
import vector_store_manager

def run_full_processing_pipeline(config: dict) -> bool:
    print_pipeline_config(config) 
    if not os.path.exists(config["docs_folder"]) or not os.listdir(config["docs_folder"]):
        print(f"Orchestrator Error: Docs folder '{config['docs_folder']}' is empty or non-existent."); return False
    if not os.path.exists(config["vector_data_dir"]): os.makedirs(config["vector_data_dir"], exist_ok=True)

    # Prepare the specific chunker_config dict to pass to prepare_documents_and_chunks
    # This ensures only relevant parameters for the selected chunker (+ general ones) are passed.
    selected_chunker_type = config["chunker_type"]
    chunker_params_for_core = {"type": selected_chunker_type} # Essential 'type' key

    # Add general chunking params not tied to a specific prefix
    # (Example: If there were any, they'd go here)

    # Add semchunk specific params if selected
    if selected_chunker_type == "semchunk":
        chunker_params_for_core["semchunk_max_tokens_chunk"] = config.get("semchunk_max_tokens_chunk", common_utils.CHUNKER_DEFAULTS["semchunk"]["max_tokens_chunk"])
        chunker_params_for_core["semchunk_overlap_percent"] = config.get("semchunk_overlap_percent", common_utils.CHUNKER_DEFAULTS["semchunk"]["overlap_percent"])
    
    # Add chonkie specific params by iterating through config keys that start with "chonkie_"
    # This is a more generic way if CLI/API args are named like "chonkie_param_name"
    # Or, be explicit for each supported chonkie type:
    elif selected_chunker_type == "chonkie_sdpm":
        for key, default_val in common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"].items():
            chunker_params_for_core[f"chonkie_{key}"] = config.get(f"chonkie_{key}", default_val) # More specific for clarity
        # Ensure specific SDPM params from config are preferred if they exist under more generic names too
        chunker_params_for_core["chonkie_embedding_model"] = config.get("chonkie_embedding_model", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["embedding_model"])
        chunker_params_for_core["chonkie_target_chunk_size"] = config.get("chonkie_target_chunk_size", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["target_chunk_size"])
        chunker_params_for_core["chonkie_similarity_threshold"] = config.get("chonkie_similarity_threshold", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["threshold"])
        chunker_params_for_core["chonkie_min_sentences"] = config.get("chonkie_min_sentences", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_sentences"])
        chunker_params_for_core["chonkie_skip_window"] = config.get("chonkie_skip_window", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["skip_window"])
        chunker_params_for_core["chonkie_sdpm_mode"] = config.get("chonkie_mode", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["mode"])
        chunker_params_for_core["chonkie_sdpm_similarity_window"] = config.get("chonkie_similarity_window", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["similarity_window"])
        chunker_params_for_core["chonkie_sdpm_min_chunk_size"] = config.get("chonkie_min_chunk_size", common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_chunk_size"])


    elif selected_chunker_type == "chonkie_neural":
        chunker_params_for_core["chonkie_neural_model"] = config.get("chonkie_neural_model", common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["model"])
        chunker_params_for_core["chonkie_neural_tokenizer"] = config.get("chonkie_neural_tokenizer", config.get("chonkie_neural_model", common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["model"])) # Often same as model
        chunker_params_for_core["chonkie_neural_stride"] = config.get("chonkie_neural_stride", common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["stride"])
        chunker_params_for_core["chonkie_neural_min_chars"] = config.get("chonkie_neural_min_chars", common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["min_chars_per_chunk"])
    # Add similar elif blocks for chonkie_semantic, chonkie_recursive, chonkie_sentence, chonkie_token
    # ensuring you pull the correct parameters from 'config' dict using their CLI/API names
    # and map them to the keys expected by data_processing_core.prepare_documents_and_chunks's new signature.

    print("\nORCHESTRATOR: Starting Document Preparation and Chunking...")
    text_chunks, chunk_ids, detailed_metadata = \
        data_processing_core.prepare_documents_and_chunks(
            docs_input_folder=config["docs_folder"], 
            chunker_selection_cfg=config["chunker_type"], # Pass the overall selection
            # Pass all potential params; prepare_documents_and_chunks will pick what it needs
            semchunk_max_tokens_cfg=config.get("semchunk_max_tokens_chunk", common_utils.CHUNKER_DEFAULTS["semchunk"]["max_tokens_chunk"]),
            semchunk_overlap_cfg=config.get("semchunk_overlap_percent", common_utils.CHUNKER_DEFAULTS["semchunk"]["overlap_percent"]),
            chonkie_emb_model_for_chunking=config.get("chonkie_embedding_model"),
            chonkie_target_chunk_size_cfg=config.get("chonkie_target_chunk_size"),
            chonkie_similarity_threshold_cfg=config.get("chonkie_similarity_threshold"),
            chonkie_min_sentences_cfg=config.get("chonkie_min_sentences"),
            chonkie_sdpm_mode_cfg=config.get("chonkie_mode"), # General chonkie_mode might map to sdpm_mode
            chonkie_sdpm_similarity_window_cfg=config.get("chonkie_similarity_window"),
            chonkie_sdpm_min_chunk_size_cfg=config.get("chonkie_min_chunk_size"),
            chonkie_sdpm_skip_window_cfg=config.get("chonkie_skip_window"),
            chonkie_semantic_mode_cfg=config.get("chonkie_mode"), # Or specific if added
            chonkie_semantic_similarity_window_cfg=config.get("chonkie_similarity_window"),
            chonkie_semantic_min_chunk_size_cfg=config.get("chonkie_min_chunk_size"),
            chonkie_neural_model_cfg=config.get("chonkie_neural_model"),
            chonkie_neural_stride_cfg=config.get("chonkie_neural_stride"),
            chonkie_neural_min_chars_cfg=config.get("chonkie_neural_min_chars"),
            chonkie_recursive_tokenizer_cfg=config.get("chonkie_basic_tokenizer"),
            chonkie_recursive_chunk_size_cfg=config.get("chonkie_basic_chunk_size"),
            chonkie_recursive_min_chars_cfg=config.get("chonkie_basic_min_chars"),
            # Add params for sentence/token chonkers here from config if they are used
            num_workers_cfg=config["num_processing_workers"], 
            enable_clf_cfg=config["enable_classification"], 
            clf_model_name_cfg=config.get("classifier_model_name"),
            clf_labels_cfg=config.get("candidate_labels"),
            clf_batch_size_cfg=config["classification_batch_size"], 
            gpu_id_cfg=config["gpu_device_id"], 
            verbose_logging_cfg=config.get("verbose", False)
        )
    if not text_chunks: print("ORCHESTRATOR: Pipeline aborted - No text chunks."); return False
    print(f"ORCHESTRATOR: Text processing complete. Total chunks: {len(text_chunks)}")

    # --- Embedding Generation (same as before) ---
    print("\nORCHESTRATOR: Starting Embedding Generation...")
    main_emb_model_name = config["embedding_model_name"]
    inferred_type, inferred_dim, inferred_metric_enum, inferred_class = embedding_module.infer_embedding_model_params(main_emb_model_name)
    final_model_type = config.get("embedding_model_type") or inferred_type
    final_default_dim = config.get("default_embedding_dim") if config.get("default_embedding_dim") is not None else inferred_dim
    pipeline_embedding_device = "cpu"
    if config["gpu_device_id"] >= 0:
        if torch.cuda.is_available(): pipeline_embedding_device = f"cuda:{config['gpu_device_id']}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): pipeline_embedding_device = "mps"
    chunk_vectors, actual_embedding_dim = embedding_module.generate_embeddings(
        text_chunks, main_emb_model_name, final_model_type, final_default_dim, inferred_class, device=pipeline_embedding_device)
    if chunk_vectors.shape[0] == 0: print("ORCHESTRATOR: Pipeline aborted - No embeddings."); return False
    print(f"ORCHESTRATOR: Embeddings generated. Dim: {actual_embedding_dim}")

    # --- Vector Store (same, but with refined metadata) ---
    print("\nORCHESTRATOR: Initializing and Saving Vector Store...")
    vsm = vector_store_manager.VectorStoreManager(config["vector_data_dir"], common_utils.VECTOR_STORE_SUBDIR_NAME)
    final_vicinity_metric = inferred_metric_enum
    if config.get("vicinity_metric"):
        try: final_vicinity_metric = getattr(Metric, config["vicinity_metric"].upper())
        except AttributeError: print(f"ORCHESTRATOR Warning: Invalid Vicinity metric. Using inferred.")
    
    store_meta_chunk_params = {"type": config["chunker_type"]}
    if config["chunker_type"] == "semchunk":
        store_meta_chunk_params.update({k.replace("semchunk_",""):v for k,v in config.items() if k.startswith("semchunk_")})
    elif config["chunker_type"].startswith("chonkie_"):
        prefix = config["chunker_type"] + "_" # e.g. "chonkie_sdpm_"
        # More careful extraction based on specific chunker type's relevant args
        if config["chunker_type"] == "chonkie_sdpm":
            relevant_keys = ["embedding_model", "target_chunk_size", "similarity_threshold", "min_sentences", "skip_window", "mode", "similarity_window", "min_chunk_size"]
            for r_key in relevant_keys: store_meta_chunk_params[r_key] = config.get(f"chonkie_{r_key}")
        elif config["chunker_type"] == "chonkie_neural":
            relevant_keys = ["neural_model", "neural_stride", "neural_min_chars"]
            for r_key in relevant_keys: store_meta_chunk_params[r_key.replace("neural_","")] = config.get(f"chonkie_{r_key}")
        # Add elif for other chonkie types...
        else: # Generic grab for other chonkie types if params follow "chonkie_X_param"
             store_meta_chunk_params.update({k.replace(config["chunker_type"]+"_", ""):v for k,v in config.items() if k.startswith(config["chunker_type"]+"_")})


    store_processing_meta = {
        "embedding_model_name": main_emb_model_name, "embedding_model_type": final_model_type,
        "embedding_model_class": inferred_class, "embedding_vector_dimension": actual_embedding_dim,
        "vicinity_metric": str(final_vicinity_metric).split('.')[-1],
        "source_docs_folder": config["docs_folder"], "pipeline_run_date": datetime.now().isoformat(),
        "classification_enabled": config["enable_classification"],
        "classifier_model_name": config.get("classifier_model_name") if config["enable_classification"] else None,
        "chunk_params": store_meta_chunk_params, # Now more structured
        "pipeline_config_summary": {k:v for k,v in config.items() if k not in ["candidate_labels", "config_file"]},
    }
    if config["enable_classification"] and config.get("candidate_labels"):
        store_processing_meta["pipeline_config_summary"]["candidate_labels_count"] = len(config["candidate_labels"])
        if len(config["candidate_labels"]) <= 10: store_processing_meta["pipeline_config_summary"]["candidate_labels_used"] = config["candidate_labels"]
        else: store_processing_meta["pipeline_config_summary"]["candidate_labels_preview"] = config["candidate_labels"][:5]

    store_created = vsm.create_and_save_store(chunk_vectors, chunk_ids, final_vicinity_metric, store_processing_meta, config.get("overwrite", False))
    if not store_created: print("ORCHESTRATOR: Pipeline aborted - Vector store not created."); return False
    
    save_orchestrator_supporting_data(config["vector_data_dir"], text_chunks, chunk_ids, detailed_metadata, chunk_vectors)
    if config.get("prepare_viz_data", False) and chunk_vectors.shape[0] > 0:
        print("\nORCHESTRATOR: Preparing Visualization Data...")
        data_processing_core.save_data_for_visualization(
            vector_data_dir=config["vector_data_dir"], chunk_vectors=chunk_vectors, metadata_list=detailed_metadata, 
            chunk_texts_list=text_chunks, output_viz_filename=config.get("viz_output_file"), 
            umap_n_neighbors=config.get("umap_neighbors"), umap_min_dist_val=config.get("umap_min_dist"), 
            umap_metric_val=config.get("umap_metric"))
    print("\n--- Pipeline Orchestrator Finished Successfully ---"); return True

def save_orchestrator_supporting_data(vector_data_dir, text_chunks, chunk_ids, detailed_metadata, chunk_vectors): # Same
    print(f"\nORCHESTRATOR: Saving supporting data files to '{vector_data_dir}'...")
    try:
        with open(os.path.join(vector_data_dir, common_utils.TEXT_CHUNKS_FILENAME), 'w', encoding='utf-8') as f: json.dump(text_chunks, f)
        with open(os.path.join(vector_data_dir, common_utils.CHUNK_IDS_FILENAME), 'w', encoding='utf-8') as f: json.dump(chunk_ids, f)
        with open(os.path.join(vector_data_dir, common_utils.DETAILED_CHUNK_METADATA_FILENAME), 'w', encoding='utf-8') as f: json.dump(detailed_metadata, f)
        if chunk_vectors.shape[0] > 0: np.save(os.path.join(vector_data_dir, common_utils.CHUNK_VECTORS_FILENAME), chunk_vectors)
        print("ORCHESTRATOR: Supporting data files saved.")
    except Exception as e: print(f"ORCHESTRATOR Error saving supporting data: {e}")

def print_pipeline_config(config: dict): # Updated for new chonkie params in config
    print(f"--- Pipeline Orchestrator Starting with Configuration ---")
    print(f"  Docs Folder: {config.get('docs_folder')}, Output Dir: {config.get('vector_data_dir')}")
    print(f"  Overwrite: {config.get('overwrite', False)}, GPU ID: {config.get('gpu_device_id')}, Verbose: {config.get('verbose',False)}")
    print(f"  Embedding Model: {config.get('embedding_model_name')}")
    chunker_type = config.get('chunker_type')
    print(f"  Chunker Type: {chunker_type.upper() if chunker_type else 'N/A'}, Workers: {config.get('num_processing_workers')}")
    if chunker_type == "semchunk":
        print(f"    Semchunk MaxTokens: {config.get('semchunk_max_tokens_chunk')}, Overlap: {config.get('semchunk_overlap_percent')}%")
    elif chunker_type == "chonkie_sdpm":
        print(f"    Chonkie (SDPM) EmbModel: {config.get('chonkie_embedding_model')}, TargetSize: {config.get('chonkie_target_chunk_size')}, SimThresh: {config.get('chonkie_similarity_threshold')}, MinSent: {config.get('chonkie_min_sentences')}, SkipWin: {config.get('chonkie_skip_window')}, Mode: {config.get('chonkie_mode')}")
    elif chunker_type == "chonkie_neural":
        print(f"    Chonkie (Neural) SegModel: {config.get('chonkie_neural_model')}, Stride: {config.get('chonkie_neural_stride')}, MinChars: {config.get('chonkie_neural_min_chars')}")
    # Add more elif for other chonkie types and their specific params
    print(f"  Classification: {'Enabled' if config.get('enable_classification') else 'Disabled'}", end="")
    if config.get('enable_classification'): print(f" (Model: {config.get('classifier_model_name')})")
    else: print()
    print(f"  Prepare Viz Data: {'Enabled' if config.get('prepare_viz_data') else 'Disabled'}")
    print("-" * 50)