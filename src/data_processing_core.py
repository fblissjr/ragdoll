# src/data_processing_core.py
# This module handles the core logic of document processing:
# - Extracting text and metadata from various file formats.
# - Chunking the extracted text using different strategies.
# - Optionally classifying the generated chunks.

import os
import json 
import multiprocessing 
from tqdm import tqdm 
from typing import List, Dict, Any, Optional, Tuple 

# RAGdoll project-specific imports
from . import ragdoll_utils
from . import ragdoll_config
from .core_logic import file_parser
from .core_logic import classification_module

# --- Attempt to import Chonkie components ---
CHONKIE_AVAILABLE = False
ChonkieChunkerInstanceType = Any 
RecursiveRules = None # Initialize; will be an alias to the class if imported.
SDPMChunker, SemanticChunker, NeuralChunker, RecursiveChunker, SentenceChunker, TokenChunker = \
    None, None, None, None, None, None # Initialize all chunker classes to None

try:
    # Try importing all required Chonkie classes directly from 'chonkie'
    # This aligns with the Chonkie documentation provided.
    from chonkie import (
        SDPMChunker as ImportedSDPMChunker, 
        SemanticChunker as ImportedSemanticChunker, 
        NeuralChunker as ImportedNeuralChunker, 
        RecursiveChunker as ImportedRecursiveChunker, 
        RecursiveRules as ImportedRecursiveRules, 
        SentenceChunker as ImportedSentenceChunker, 
        TokenChunker as ImportedTokenChunker
    )
    
    # Assign to module-level variables if import was successful
    SDPMChunker, SemanticChunker, NeuralChunker = ImportedSDPMChunker, ImportedSemanticChunker, ImportedNeuralChunker
    RecursiveChunker, SentenceChunker, TokenChunker = ImportedRecursiveChunker, ImportedSentenceChunker, ImportedTokenChunker
    RecursiveRules = ImportedRecursiveRules
    
    CHONKIE_AVAILABLE = True
    print("Data Processing Core: All required Chonkie components (including RecursiveRules) imported successfully from top-level 'chonkie'.")

except ImportError as e_chonk:
    print(f"Data Processing Core: CRITICAL WARNING - Failed to import core Chonkie components: {e_chonk}. "
          "Ensure 'chonkie' is correctly installed with all necessary extras (e.g., 'pip install \"chonkie[all]\"'). "
          "Chunking functionality will be severely limited or non-functional.")
    # Define dummy classes if main import fails, to prevent NameErrors elsewhere,
    # but the application won't function correctly for Chonkie-dependent parts.
    # These will be used if CHONKIE_AVAILABLE remains False.
    class SDPMChunker: pass
    class SemanticChunker: pass
    class NeuralChunker: pass
    class RecursiveChunker: pass
    # RecursiveRules remains None
    class SentenceChunker: pass
    class TokenChunker: pass
    
# Global variable to hold the initialized chunker instance within each worker process.
WORKER_CHUNKER_INSTANCE: Optional[ChonkieChunkerInstanceType] = None 

def _init_worker_chunker(chunker_config_from_orchestrator: Dict[str, Any]):
    """
    Initializes the appropriate Chonkie chunker instance within a worker process.
    """
    global WORKER_CHUNKER_INSTANCE, CHONKIE_AVAILABLE, RecursiveRules # Allow modification
    global SDPMChunker, SemanticChunker, NeuralChunker, RecursiveChunker, SentenceChunker, TokenChunker # Reference module-level (potentially dummy) classes
    
    chunker_config_for_init = chunker_config_from_orchestrator.copy()
    chunker_type_name = chunker_config_for_init.pop("type", None)
    if not chunker_type_name:
        print("Worker Init Error: 'type' key missing in chunker_config. Cannot initialize.")
        WORKER_CHUNKER_INSTANCE = None
        return

    params_for_constructor = chunker_config_for_init 

    print(f"Worker Init: Attempting Chonkie chunker type '{chunker_type_name}' with params: {params_for_constructor}")
    WORKER_CHUNKER_INSTANCE = None

    if not CHONKIE_AVAILABLE:
        print(f"Worker Init Error: Chonkie library core components are not available (CHONKIE_AVAILABLE=False). Cannot initialize chunker '{chunker_type_name}'.")
        return
        
    # --- Tokenizer Resolution for relevant chunkers ---
    tokenizer_key_to_check = None
    if chunker_type_name in ["chonkie_sentence", "chonkie_recursive"]:
        tokenizer_key_to_check = "tokenizer_or_token_counter"
    elif chunker_type_name == "chonkie_token":
        tokenizer_key_to_check = "tokenizer"
    
    if tokenizer_key_to_check and tokenizer_key_to_check in params_for_constructor:
        tokenizer_setting = params_for_constructor[tokenizer_key_to_check]
        if isinstance(tokenizer_setting, str):
            if tokenizer_setting == "ragdoll_utils.BGE_TOKENIZER_INSTANCE":
                params_for_constructor[tokenizer_key_to_check] = ragdoll_utils.BGE_TOKENIZER_INSTANCE
            elif tokenizer_setting == "ragdoll_utils.count_tokens_robustly":
                 params_for_constructor[tokenizer_key_to_check] = ragdoll_utils.count_tokens_robustly

    # --- Chunker Instantiation ---
    try:
        chunker_class_to_init = None
        is_recipe_based_init = False

        if chunker_type_name == "chonkie_sdpm":
            chunker_class_to_init = SDPMChunker
        elif chunker_type_name == "chonkie_semantic":
            chunker_class_to_init = SemanticChunker
        elif chunker_type_name == "chonkie_neural":
            chunker_class_to_init = NeuralChunker
        elif chunker_type_name == "chonkie_recursive":
            chunker_class_to_init = RecursiveChunker
            recipe_name_for_recursive = params_for_constructor.get("rules") # Orchestrator puts recipe name here
            
            if isinstance(recipe_name_for_recursive, str):
                lang_for_recipe = params_for_constructor.pop("lang", "en") 
                # Remove 'rules' and 'lang' as they are handled by from_recipe
                params_for_constructor.pop("rules", None)
                
                print(f"  Worker Info (chonkie_recursive): Using from_recipe('{recipe_name_for_recursive}', lang='{lang_for_recipe}') with other params: {params_for_constructor}")
                WORKER_CHUNKER_INSTANCE = RecursiveChunker.from_recipe(
                    recipe_name_for_recursive, 
                    lang=lang_for_recipe, 
                    **params_for_constructor 
                )
                is_recipe_based_init = True
            else: # No recipe string for 'rules', use direct constructor
                if RecursiveRules and not isinstance(params_for_constructor.get("rules"), RecursiveRules):
                    print(f"  Worker Info (chonkie_recursive): 'rules' is '{params_for_constructor.get('rules')}'. Defaulting to RecursiveRules().")
                    params_for_constructor["rules"] = RecursiveRules()
                elif not RecursiveRules:
                    print("  Worker Warning (chonkie_recursive): RecursiveRules class not available. Chonkie will use internal default for 'rules'.")
                    params_for_constructor.pop("rules", None) # Rely on Chonkie's constructor default for rules

        elif chunker_type_name == "chonkie_sentence":
            chunker_class_to_init = SentenceChunker
        elif chunker_type_name == "chonkie_token":
            chunker_class_to_init = TokenChunker
            tokenizer_val = params_for_constructor.get("tokenizer")
            if hasattr(tokenizer_val, "is_fast") and tokenizer_val.is_fast: # HF Fast Tokenizer
                try:
                    from chonkie.tokenizers import Tokenizer as ChonkieTokenizerWrapper
                    print("  Worker Info (chonkie_token): Wrapping RAGDOLL BGE Tokenizer with ChonkieTokenizerWrapper.")
                    params_for_constructor["tokenizer"] = ChonkieTokenizerWrapper(tokenizer_val)
                except ImportError:
                    print("  Worker Warning (chonkie_token): Could not import ChonkieTokenizerWrapper. Passing tokenizer as is.")
        else:
            print(f"Worker Init Error: Chonkie chunker type '{chunker_type_name}' is not handled.")
            return

        if not is_recipe_based_init and chunker_class_to_init:
            # Ensure the class is not None (i.e., was actually imported)
            if chunker_class_to_init is not None and not (isinstance(chunker_class_to_init, type) and issubclass(chunker_class_to_init, (SDPMChunker if SDPMChunker else object, SemanticChunker if SemanticChunker else object, NeuralChunker if NeuralChunker else object, RecursiveChunker if RecursiveChunker else object, SentenceChunker if SentenceChunker else object, TokenChunker if TokenChunker else object))):
                 # This check is a bit verbose due to dummy classes; simplifies if imports always succeed
                 pass


            if chunker_class_to_init is None: # If it's still None, it means the dummy class is being used.
                 print(f"  Worker Error ({chunker_type_name}): Corresponding Chonkie class was not imported. Cannot initialize.")
            else:
                WORKER_CHUNKER_INSTANCE = chunker_class_to_init(**params_for_constructor)
        
    except ImportError as e_imp:
        print(f"Worker Init Error: Missing Chonkie dependencies for '{chunker_type_name}'. Error: {e_imp}")
        WORKER_CHUNKER_INSTANCE = None
    except Exception as e_init:
        print(f"Worker Init Error: Failed to initialize Chonkie chunker '{chunker_type_name}' with params {params_for_constructor}. Error: {e_init}")
        WORKER_CHUNKER_INSTANCE = None
    
    if WORKER_CHUNKER_INSTANCE: 
        print(f"Worker Chunker '{chunker_type_name}' initialized successfully.")
    else: 
        print(f"Worker Chunker '{chunker_type_name}' FAILED to initialize.")

def _chunk_doc_content_worker_entry(args_tuple: Tuple[str, Dict[str, Any], str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    doc_content, base_doc_part_metadata, doc_part_id_prefix, chunker_config_from_orchestrator = args_tuple
    if WORKER_CHUNKER_INSTANCE is None:
        chunker_type_for_fallback = chunker_config_from_orchestrator.get('type', 'unknown_worker_type')
        default_chunk_size_for_fallback = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_for_fallback, {}).get('chunk_size', 256)
        fallback_max_char_len = default_chunk_size_for_fallback * 5 
        chunk_text_content = doc_content[:fallback_max_char_len]
        chunk_vicinity_id = f"{doc_part_id_prefix}_chunk_0_init_fail" 
        display_source_name_for_chunk = ragdoll_utils.generate_display_source_name(base_doc_part_metadata, chunk_index=0) 
        metadata_for_fallback_chunk = {
            **base_doc_part_metadata, 
            "chunk_order_in_doc_part": 0, 
            "vicinity_item_id": chunk_vicinity_id, 
            "display_source_name": display_source_name_for_chunk, 
            "chunking_error": f"Worker chunker '{chunker_type_for_fallback}' was not initialized. Using fallback chunk."
        }
        return [chunk_text_content] if chunk_text_content.strip() else [], \
               [chunk_vicinity_id] if chunk_text_content.strip() else [], \
               [metadata_for_fallback_chunk] if chunk_text_content.strip() else []

    text_chunks_for_doc_part: List[str] = []
    chunk_ids_for_doc_part: List[str] = []
    detailed_meta_for_doc_part: List[Dict[str, Any]] = []
    processed_chunk_texts_list: List[str] = []
    chunker_type_name = chunker_config_from_orchestrator.get("type", "unknown_chonkie_type")
    
    try:
        chonkie_output_texts = WORKER_CHUNKER_INSTANCE.chunk(doc_content) 
        if chonkie_output_texts and isinstance(chonkie_output_texts, list):
            # Ensure all items are strings, as 'return_type="texts"' is set by orchestrator
            processed_chunk_texts_list = [str(text) for text in chonkie_output_texts if isinstance(text, (str, bytes)) and str(text).strip()]
        else: 
            processed_chunk_texts_list = []
            if chonkie_output_texts:
                print(f"  Worker Warning (chunking): Unexpected output from chunker '{chunker_type_name}': {type(chonkie_output_texts)}. Expected list of strings.")
    except Exception as e_chunk:
        print(f"Worker Error: Chunking document part '{doc_part_id_prefix}' with chunker '{chunker_type_name}' failed: {e_chunk}")
        default_chunk_size_on_error = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_name, {}).get('chunk_size', 256)
        fallback_max_char_len_on_error = default_chunk_size_on_error * 5 
        processed_chunk_texts_list = [doc_content[:fallback_max_char_len_on_error]] 
        base_doc_part_metadata["chunking_error"] = f"Chunking failed for {chunker_type_name}: {str(e_chunk)}"

    for idx, text_item_content in enumerate(processed_chunk_texts_list):
        if text_item_content and text_item_content.strip(): 
            chunk_vicinity_id_gen = f"{doc_part_id_prefix}_chunk_{idx}" 
            text_chunks_for_doc_part.append(text_item_content)
            chunk_ids_for_doc_part.append(chunk_vicinity_id_gen)
            display_source_name_for_this_chunk = ragdoll_utils.generate_display_source_name(base_doc_part_metadata, chunk_index=idx)
            meta_for_this_chunk = {
                **base_doc_part_metadata, 
                "chunk_order_in_doc_part": idx, 
                "vicinity_item_id": chunk_vicinity_id_gen, 
                "display_source_name": display_source_name_for_this_chunk
            }
            detailed_meta_for_doc_part.append(meta_for_this_chunk)       
    
    if not text_chunks_for_doc_part and doc_content and doc_content.strip():
        default_chunk_size_final_fb = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_name, {}).get('chunk_size', 256)
        fallback_max_char_len_final_fb = default_chunk_size_final_fb * 5
        chunk_text_content_final_fb = doc_content[:fallback_max_char_len_final_fb]
        chunk_vicinity_id_final_fb = f"{doc_part_id_prefix}_chunk_0_full_content_fallback"
        text_chunks_for_doc_part.append(chunk_text_content_final_fb)
        chunk_ids_for_doc_part.append(chunk_vicinity_id_final_fb)
        display_source_name_final_fb = ragdoll_utils.generate_display_source_name(base_doc_part_metadata, chunk_index=0) 
        meta_final_fb = {
            **base_doc_part_metadata, 
            "chunk_order_in_doc_part": 0, 
            "vicinity_item_id": chunk_vicinity_id_final_fb, 
            "display_source_name": display_source_name_final_fb, 
            "chunking_notes": "Full content fallback as no other chunks were generated."
        }
        if "chunking_error" not in meta_final_fb: 
             meta_final_fb["chunking_error"] = "No chunks produced by selected chunker; using full content fallback."
        detailed_meta_for_doc_part.append(meta_final_fb)
        
    return text_chunks_for_doc_part, chunk_ids_for_doc_part, detailed_meta_for_doc_part

def prepare_documents_and_chunks(
    docs_input_folder: str, 
    chunker_config: Dict[str, Any], 
    num_workers_cfg: int, 
    enable_clf_cfg: bool, 
    clf_model_name_cfg: Optional[str], 
    clf_labels_cfg: Optional[List[str]], 
    clf_batch_size_cfg: int, 
    gpu_id_cfg: int, 
    verbose_logging_cfg: bool
) -> tuple[List[str], List[str], List[Dict[str, Any]]]:
    extracted_doc_parts_with_meta: List[Tuple[str, Dict[str, Any], str]] = [] 
    file_paths_to_process: List[str] = []
    for root_dir, sub_dirs, files_in_dir in os.walk(docs_input_folder):
        sub_dirs[:] = [d for d in sub_dirs if not d.startswith('.') and not d.lower() == '__macosx'] 
        for f_name in files_in_dir:
            if not f_name.startswith('.') and not f_name.lower().endswith(('.tmp', '.temp', '.ds_store')): 
                file_paths_to_process.append(os.path.join(root_dir, f_name))

    if not file_paths_to_process: 
        print(f"Data Processing Core: No processable files found in '{docs_input_folder}'. Returning empty lists.")
        return [], [], []
    print(f"PHASE 1: Text Extraction - Found {len(file_paths_to_process)} candidate files in '{docs_input_folder}'.")

    for fpath in tqdm(file_paths_to_process, desc="Extracting File Contents", disable=not verbose_logging_cfg):
        relative_fpath = os.path.relpath(fpath, docs_input_folder)
        sanitized_base_filename_id = ragdoll_utils.sanitize_filename_for_id(relative_fpath)
        doc_parts_from_single_file = file_parser.extract_documents_from_file(fpath, relative_fpath)
        
        for doc_part_idx, (content_str, metadata_dict) in enumerate(doc_parts_from_single_file):
            id_suffix_components: List[str] = []
            id_component_keys = ["page_number", "sheet_name", "row_index", "line_number", "json_path", "epub_item_id", "epub_item_name"]
            for key_for_id_part in id_component_keys:
                if key_for_id_part in metadata_dict and metadata_dict[key_for_id_part] is not None: 
                    id_suffix_components.append(
                        f"{key_for_id_part.split('_')[0][:3]}" 
                        f"{ragdoll_utils.sanitize_filename_for_id(str(metadata_dict[key_for_id_part]))[:15]}"
                    )
            current_doc_part_id_prefix = sanitized_base_filename_id
            if id_suffix_components:
                current_doc_part_id_prefix = f"{sanitized_base_filename_id}_{'_'.join(id_suffix_components)}"
            elif len(doc_parts_from_single_file) > 1: 
                current_doc_part_id_prefix = f"{sanitized_base_filename_id}_docpart{doc_part_idx}"
            current_doc_part_id_prefix = ragdoll_utils.sanitize_filename_for_id(current_doc_part_id_prefix) 
            if content_str and content_str.strip(): 
                metadata_dict["doc_part_id_prefix"] = current_doc_part_id_prefix 
                extracted_doc_parts_with_meta.append((content_str, metadata_dict, current_doc_part_id_prefix))
    
    if not extracted_doc_parts_with_meta: 
        print("Data Processing Core: No text content was extracted from any files. Returning empty lists.")
        return [], [], []

    print(f"\nPHASE 2: Chunking {len(extracted_doc_parts_with_meta)} document parts "
          f"(Using {num_workers_cfg} workers, Chunker: {chunker_config.get('type','N/A').upper()})...")
    tasks_for_chunking_pool = [
        (part_content, part_meta, part_id_prefix, chunker_config) 
        for part_content, part_meta, part_id_prefix in extracted_doc_parts_with_meta
    ]
    all_text_chunks: List[str] = []
    all_chunk_ids: List[str] = []
    all_detailed_meta: List[Dict[str, Any]] = []
    pool_initializer_func = _init_worker_chunker 
    pool_initargs_tuple = (chunker_config,) 
    actual_num_workers_to_use = min(num_workers_cfg, len(tasks_for_chunking_pool)) if tasks_for_chunking_pool else 1
    actual_num_workers_to_use = max(1, actual_num_workers_to_use) 
    chunking_results = [] # Initialize to ensure it's defined

    if actual_num_workers_to_use > 1 and len(tasks_for_chunking_pool) > 1:
        mp_context_name = "spawn" if os.name != 'nt' else None
        print(f"Data Processing Core: Attempting parallel chunking with {actual_num_workers_to_use} workers (context: {mp_context_name or 'default'}).")
        try:
            mp_context = multiprocessing.get_context(mp_context_name)
            with mp_context.Pool(processes=actual_num_workers_to_use, 
                                initializer=pool_initializer_func, 
                                initargs=pool_initargs_tuple) as pool:
                chunking_results = list(tqdm(
                    pool.imap_unordered(_chunk_doc_content_worker_entry, tasks_for_chunking_pool), 
                    total=len(tasks_for_chunking_pool), 
                    desc="Parallel Chunking"
                ))
        except Exception as e_pool:
            print(f"Data Processing Core: Error initializing multiprocessing pool with context '{mp_context_name}': {e_pool}")
            print("Data Processing Core: Falling back to sequential chunking.")
            actual_num_workers_to_use = 1 
    
    if actual_num_workers_to_use <= 1 or not chunking_results:
        if actual_num_workers_to_use > 1 and not chunking_results: 
            print("Data Processing Core: Parallel chunking setup failed or yielded no results, proceeding sequentially.")
        else:
             print("Data Processing Core: Using sequential chunking.")
        pool_initializer_func(*pool_initargs_tuple) 
        chunking_results = [
            _chunk_doc_content_worker_entry(task_args) 
            for task_args in tqdm(tasks_for_chunking_pool, desc="Sequential Chunking")
        ]
    
    for text_chunks_list, chunk_ids_list, detailed_meta_list in chunking_results:
        all_text_chunks.extend(text_chunks_list)
        all_chunk_ids.extend(chunk_ids_list)
        all_detailed_meta.extend(detailed_meta_list)
    
    if not all_text_chunks: 
        print("Data Processing Core: No chunks were generated after the chunking phase. Returning empty lists.")
        return [], [], []
    print(f"Total chunks generated after chunking phase: {len(all_text_chunks)}")

    # --- Phase 3: Optional Chunk Classification ---
    if enable_clf_cfg and clf_model_name_cfg and all_text_chunks:
        print(f"\nPHASE 3: Classifying {len(all_text_chunks)} chunks using '{clf_model_name_cfg}'...")
        
        classifier_instance = classification_module.initialize_classifier(
            model_name_or_path=clf_model_name_cfg, 
            device_id=gpu_id_cfg,
            # Pass the batch_size from RAGdoll config to initialize the HF pipeline
            batch_size_for_pipeline_init=clf_batch_size_cfg 
        )
        
        if classifier_instance:
            actual_classification_labels = clf_labels_cfg if clf_labels_cfg else ragdoll_config.DEFAULT_CLASSIFICATION_LABELS
            
            # Call classify_chunks_batch, also passing the batch_size for the call itself
            classification_outputs = classification_module.classify_chunks_batch(
                chunks_to_classify=all_text_chunks, 
                classifier=classifier_instance, 
                candidate_labels=actual_classification_labels,
                batch_size_for_call=clf_batch_size_cfg, # Pass batch_size here too
                multi_label_classification=True 
            )
        # if classifier_instance:
        #     actual_classification_labels = clf_labels_cfg if clf_labels_cfg else ragdoll_config.DEFAULT_CLASSIFICATION_LABELS
            
        #     # Call classify_chunks_batch (it no longer takes batch_size_for_call)
        #     classification_outputs = classification_module.classify_chunks_batch(
        #         chunks_to_classify=all_text_chunks, 
        #         classifier=classifier_instance, 
        #         candidate_labels=actual_classification_labels,
        #         multi_label_classification=False 
        #     )
            
            if len(classification_outputs) == len(all_detailed_meta): 
                for i, class_output_item_dict in enumerate(classification_outputs): 
                    if class_output_item_dict: 
                        all_detailed_meta[i].update(class_output_item_dict)
            else: 
                print(f"Data Processing Core Warning: Classification output count ({len(classification_outputs)}) "
                      f"does not match metadata count ({len(all_detailed_meta)}). Metadata might be incomplete.")
        else:
            print("  Classification skipped because the classifier model could not be initialized.")
            for i in range(len(all_detailed_meta)):
                all_detailed_meta[i].update({"classification_status": "skipped_classifier_init_failed"})
                
    elif enable_clf_cfg and not clf_model_name_cfg:
        print("  Classification was enabled, but no classifier model name was provided. Skipping classification.")
        for i in range(len(all_detailed_meta)):
            all_detailed_meta[i].update({"classification_status": "skipped_no_model_name"})
            
    return all_text_chunks, all_chunk_ids, all_detailed_meta