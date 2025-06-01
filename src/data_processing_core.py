# src/data_processing_core.py
# Core module for orchestrating document preparation, including text extraction (via file_parser),
# chunking (via Chonkie workers), and optional classification (via classification_module).

import os
import json
import multiprocessing
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

# Project-specific imports
from . import ragdoll_utils # Utility functions
from . import ragdoll_config # Default configurations
# Import from new sub-modules if they are in a 'core_logic' subfolder
from .core_logic import file_parser # For text extraction from files
from .core_logic import classification_module # For classifying chunks

# Chonkie library for advanced chunking
CHONKIE_AVAILABLE = False
ChonkieChunkerInstanceType = Any 

try:
    from chonkie import (
        SDPMChunker, SemanticChunker, NeuralChunker, 
        RecursiveChunker, SentenceChunker, TokenChunker
    )
    CHONKIE_AVAILABLE = True
    # print("Data Processing Core: Chonkie chunkers imported.") # Already printed by individual modules
except ImportError as e_chonk:
    print(f"Data Processing Core: Warning - Chonkie chunkers might be unavailable: {e_chonk}")
    class SDPMChunker:pass; class SemanticChunker:pass; class NeuralChunker:pass; 
    class RecursiveChunker:pass; class SentenceChunker:pass; class TokenChunker:pass;
    
# Global variable for worker chunker instance
WORKER_CHUNKER_INSTANCE: Optional[ChonkieChunkerInstanceType] = None 

def _init_worker_chunker(chunker_config_from_orchestrator: Dict[str, Any]):
    """
    Initializes the Chonkie chunker instance within a worker process.
    (Content of this function is the same as provided in your Step 2, just ensure imports are correct)
    """
    global WORKER_CHUNKER_INSTANCE
    chunker_config_for_init = chunker_config_from_orchestrator.copy()
    chunker_type = chunker_config_for_init.pop("type")
    params_for_constructor = chunker_config_for_init 

    print(f"Worker Init: Attempting Chonkie chunker type '{chunker_type}' with params: {params_for_constructor}")
    WORKER_CHUNKER_INSTANCE = None

    if not CHONKIE_AVAILABLE:
        print(f"Worker Init Error: Chonkie library not available. Cannot initialize '{chunker_type}'.")
        return

    try:
        if chunker_type == "chonkie_sdpm":
            params_for_constructor.setdefault("device", "cpu")
            WORKER_CHUNKER_INSTANCE = SDPMChunker(**params_for_constructor)
        elif chunker_type == "chonkie_semantic":
            params_for_constructor.setdefault("device", "cpu")
            WORKER_CHUNKER_INSTANCE = SemanticChunker(**params_for_constructor)
        elif chunker_type == "chonkie_neural":
            params_for_constructor.setdefault("device_map", "cpu")
            WORKER_CHUNKER_INSTANCE = NeuralChunker(**params_for_constructor)
        elif chunker_type == "chonkie_recursive":
            tokenizer_setting = params_for_constructor.get("tokenizer_or_token_counter")
            if isinstance(tokenizer_setting, str) and tokenizer_setting == "ragdoll_utils.BGE_TOKENIZER_INSTANCE":
                params_for_constructor["tokenizer_or_token_counter"] = ragdoll_utils.BGE_TOKENIZER_INSTANCE
            WORKER_CHUNKER_INSTANCE = RecursiveChunker(**params_for_constructor)
        elif chunker_type == "chonkie_sentence":
            tokenizer_setting = params_for_constructor.get("tokenizer_or_token_counter")
            if isinstance(tokenizer_setting, str) and tokenizer_setting == "ragdoll_utils.BGE_TOKENIZER_INSTANCE":
                params_for_constructor["tokenizer_or_token_counter"] = ragdoll_utils.BGE_TOKENIZER_INSTANCE
            WORKER_CHUNKER_INSTANCE = SentenceChunker(**params_for_constructor)
        elif chunker_type == "chonkie_token":
            WORKER_CHUNKER_INSTANCE = TokenChunker(**params_for_constructor)
        else:
            print(f"Worker Init Error: Chonkie type '{chunker_type}' not handled in _init_worker_chunker.")
    except Exception as e:
        print(f"Worker Init Error: Failed to initialize Chonkie chunker '{chunker_type}' with {params_for_constructor}: {e}")
        WORKER_CHUNKER_INSTANCE = None
    
    if WORKER_CHUNKER_INSTANCE: print(f"Worker Chunker '{chunker_type}' initialized.")
    else: print(f"Worker Chunker '{chunker_type}' FAILED to initialize.")


def _chunk_doc_content_worker_entry(args_tuple: Tuple[str, Dict[str, Any], str, Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Worker function for multiprocessing pool to chunk a single document part.
    (Content of this function is the same as provided in your Step 2, ensure ragdoll_utils.generate_display_source_name is used)
    """
    doc_content, base_doc_part_metadata, doc_part_id_prefix, chunker_params_from_orchestrator = args_tuple
    
    if WORKER_CHUNKER_INSTANCE is None:
        chunker_type_for_fallback = chunker_params_from_orchestrator.get('type', 'unknown_worker_type')
        default_size = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_for_fallback, {}).get('chunk_size', 256)
        fallback_max_len = default_size * 5
        chunk_text = doc_content[:fallback_max_len]
        vic_id = f"{doc_part_id_prefix}_chunk_0_init_fail"
        disp_name = ragdoll_utils.generate_display_source_name(base_doc_part_metadata, 0)
        meta = {**base_doc_part_metadata, "chunk_order_in_doc_part":0, "vicinity_item_id":vic_id, "display_source_name":disp_name, "chunking_error":f"Worker chunker '{chunker_type_for_fallback}' not init."}
        return [chunk_text] if chunk_text.strip() else [], [vic_id] if chunk_text.strip() else [], [meta] if chunk_text.strip() else []

    text_chunks_for_doc_part, chunk_ids_for_doc_part, detailed_meta_for_doc_part = [], [], []
    chunk_texts_list: List[str] = []
    chunker_type_name = chunker_params_from_orchestrator["type"]
    
    try:
        if not chunker_type_name.startswith("chonkie_"):
             raise ValueError(f"Unsupported chunker '{chunker_type_name}'. Only 'chonkie_' types expected.")

        if chunker_type_name == "chonkie_recursive" and base_doc_part_metadata.get("content_type") == "markdown":
            print(f"  Info (Worker): Chunking Markdown for '{doc_part_id_prefix}' with {chunker_type_name}.")
        
        chonkie_output = WORKER_CHUNKER_INSTANCE.chunk(doc_content) 
        
        if chonkie_output:
            if isinstance(chonkie_output[0], str): 
                chunk_texts_list = [text for text in chonkie_output if text.strip()]
            else: 
                chunk_texts_list = [getattr(c, 'text', str(c)).strip() for c in chonkie_output if hasattr(c, 'text') and getattr(c, 'text', '').strip()]
        else: 
            chunk_texts_list = []
            print(f"  Info (Worker): Chonkie '{chunker_type_name}' returned no chunks for '{doc_part_id_prefix}'.")

    except Exception as e_chunk:
        print(f"Worker Error: Chunking '{doc_part_id_prefix}' with {chunker_type_name} failed: {e_chunk}")
        default_size = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_name, {}).get('chunk_size', 256)
        fallback_max_len = default_size * 5 
        chunk_texts_list = [doc_content[:fallback_max_len]]
        base_doc_part_metadata["chunking_error"] = str(e_chunk)

    for idx, text_item in enumerate(chunk_texts_list):
        if text_item.strip():
            vic_id = f"{doc_part_id_prefix}_chunk_{idx}"
            text_chunks_for_doc_part.append(text_item)
            chunk_ids_for_doc_part.append(vic_id)
            disp_name_chunk = ragdoll_utils.generate_display_source_name(base_doc_part_metadata, idx)
            meta_for_this_chunk = {**base_doc_part_metadata, "chunk_order_in_doc_part":idx, "vicinity_item_id":vic_id, "display_source_name": disp_name_chunk}
            detailed_meta_for_doc_part.append(meta_for_this_chunk)       
    
    if not text_chunks_for_doc_part and doc_content.strip():
        print(f"  Warning (Worker): No chunks from '{doc_part_id_prefix}'. Creating fallback.")
        default_size = ragdoll_config.CHUNKER_DEFAULTS.get(chunker_type_name, {}).get('chunk_size', 256)
        fallback_max_len = default_size * 5
        chunk_text_fb = doc_content[:fallback_max_len]
        vic_id_fb = f"{doc_part_id_prefix}_chunk_0_fullfb"
        text_chunks_for_doc_part.append(chunk_text_fb)
        chunk_ids_for_doc_part.append(vic_id_fb)
        disp_name_fb = ragdoll_utils.generate_display_source_name(base_doc_part_metadata, 0)
        meta_fb = {**base_doc_part_metadata, "chunk_order_in_doc_part":0, "vicinity_item_id": vic_id_fb, "display_source_name": disp_name_fb, "chunking_notes": "Full content fallback."}
        detailed_meta_for_doc_part.append(meta_fb)
        
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
) -> tuple[list, list, list]:
    """
    Main coordinating function for document processing.
    1. Finds all processable files.
    2. Extracts text and metadata from each file using file_parser module.
    3. Chunks the extracted document parts in parallel using worker processes.
    4. Optionally classifies the generated chunks using classification_module.
    """
    
    # --- PHASE 1: File Discovery and Text Extraction ---
    extracted_doc_parts_with_meta = [] 
    
    file_paths_to_process = []
    for root_dir, sub_dirs, files_in_dir in os.walk(docs_input_folder):
        sub_dirs[:] = [d for d in sub_dirs if not d.startswith('.') and not d.lower() == '__macosx'] 
        for f_name in files_in_dir:
            if not f_name.startswith('.') and not f_name.lower().endswith(('.tmp', '.temp', '.ds_store')): 
                file_paths_to_process.append(os.path.join(root_dir, f_name))

    if not file_paths_to_process: 
        print(f"Data Processing Core: No processable files found in '{docs_input_folder}'."); return [], [], []
    print(f"PHASE 1: Text Extraction - Found {len(file_paths_to_process)} candidate files.")

    for fpath in tqdm(file_paths_to_process, desc="Extracting File Contents", disable=not verbose_logging_cfg):
        relative_fpath = os.path.relpath(fpath, docs_input_folder)
        sanitized_base_filename_id = ragdoll_utils.sanitize_filename_for_id(relative_fpath)
        
        # Delegate to file_parser module for actual extraction
        doc_parts_from_single_file = file_parser.extract_documents_from_file(fpath, relative_fpath)
        
        for doc_part_idx, (content, meta) in enumerate(doc_parts_from_single_file):
            id_suffix_components = []
            for key_for_id in ["page_number", "sheet_name", "row_index", "line_number", "json_path", "epub_item_id", "epub_item_name"]:
                if key_for_id in meta and meta[key_for_id] is not None: 
                    id_suffix_components.append(f"{key_for_id.split('_')[0][:3]}{ragdoll_utils.sanitize_filename_for_id(str(meta[key_for_id]))[:15]}")
            
            current_doc_part_id_prefix = sanitized_base_filename_id
            if id_suffix_components:
                current_doc_part_id_prefix = f"{sanitized_base_filename_id}_{'_'.join(id_suffix_components)}"
            elif len(doc_parts_from_single_file) > 1: 
                current_doc_part_id_prefix = f"{sanitized_base_filename_id}_docpart{doc_part_idx}"
            
            current_doc_part_id_prefix = ragdoll_utils.sanitize_filename_for_id(current_doc_part_id_prefix) 
            
            if content and content.strip(): 
                meta["doc_part_id_prefix"] = current_doc_part_id_prefix 
                extracted_doc_parts_with_meta.append((content, meta, current_doc_part_id_prefix))
    
    if not extracted_doc_parts_with_meta: 
        print("Data Processing Core: No text content extracted."); return [], [], []

    # --- PHASE 2: Chunking ---
    print(f"\nPHASE 2: Chunking {len(extracted_doc_parts_with_meta)} doc parts (Workers: {num_workers_cfg}, Chunker: {chunker_config.get('type','N/A').upper()})...")
    tasks_for_chunking_pool = [(pc, pm, pid, chunker_config) for pc, pm, pid in extracted_doc_parts_with_meta]
    
    all_text_chunks, all_chunk_ids, all_detailed_meta = [], [], []
    
    pool_initializer_func = _init_worker_chunker 
    pool_initargs_tuple = (chunker_config,)
    actual_num_workers = min(num_workers_cfg, len(tasks_for_chunking_pool)) if tasks_for_chunking_pool else 1
    
    if actual_num_workers > 1 and len(tasks_for_chunking_pool) > 1:
        with multiprocessing.Pool(processes=actual_num_workers, initializer=pool_initializer_func, initargs=pool_initargs_tuple) as pool:
            chunking_results = list(tqdm(pool.imap_unordered(_chunk_doc_content_worker_entry, tasks_for_chunking_pool), 
                                         total=len(tasks_for_chunking_pool), desc="Parallel Chunking"))
    else: 
        pool_initializer_func(*pool_initargs_tuple)
        chunking_results = [_chunk_doc_content_worker_entry(task_args) for task_args in tqdm(tasks_for_chunking_pool, desc="Sequential Chunking")]
    
    for tc_list, cid_list, dm_list in chunking_results:
        all_text_chunks.extend(tc_list); all_chunk_ids.extend(cid_list); all_detailed_meta.extend(dm_list)
    
    if not all_text_chunks: 
        print("Data Processing Core: No chunks generated."); return [], [], []
    print(f"Total chunks generated: {len(all_text_chunks)}")

    # --- PHASE 3: Optional Classification ---
    if enable_clf_cfg and clf_model_name_cfg and all_text_chunks:
        print(f"\nPHASE 3: Classifying {len(all_text_chunks)} chunks using '{clf_model_name_cfg}'...")
        # Delegate to classification_module
        classifier_instance = classification_module.initialize_classifier(clf_model_name_cfg, device_id=gpu_id_cfg)
        if classifier_instance:
            actual_clf_labels = clf_labels_cfg if clf_labels_cfg else ragdoll_config.DEFAULT_CLASSIFICATION_LABELS
            classification_outputs = classification_module.classify_chunks_batch(
                all_text_chunks, classifier_instance, actual_clf_labels, clf_batch_size_cfg
            )
            if len(classification_outputs) == len(all_detailed_meta): 
                for i, class_output_item in enumerate(classification_outputs): 
                    if class_output_item: all_detailed_meta[i].update(class_output_item)
            else: 
                print(f"Data Processing Core Warning: Classification output count mismatch. Results: {len(classification_outputs)}, Metadata: {len(all_detailed_meta)}.")
        else:
            print("  Classification skipped (classifier init failed).")
            for i in range(len(all_detailed_meta)):
                all_detailed_meta[i].update({"classification_status": "skipped_classifier_init_failed"})

    return all_text_chunks, all_chunk_ids, all_detailed_meta