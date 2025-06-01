# data_processing_core.py
import os
import json
import re
import polars as pl
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from typing import List, Dict, Any, Optional, Tuple, Callable

from pypdf import PdfReader
from docx import Document as DocxDocument
# import openpyxl # Not directly used if Polars handles xlsx with its engine
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
import numpy

import common_utils

SEMCHUNK_AVAILABLE = False
CHONKIE_AVAILABLE = False
# Define types for Chonkie chunkers more generically for the worker
ChonkieChunkerInstanceType = Any 

try:
    from semchunk import chunkerify as semchunk_main_chunkerify_func
    SEMCHUNK_AVAILABLE = True
except ImportError:
    def semchunk_main_chunkerify_func(*_args, **_kwargs): raise NotImplementedError("semchunk not imported")

try:
    from chonkie import (
        SDPMChunker, SemanticChunker, NeuralChunker, 
        RecursiveChunker, SentenceChunker, TokenChunker
    )
    CHONKIE_AVAILABLE = True
    print("Data Processing Core: All listed Chonkie chunkers imported.")
except ImportError as e_chonk:
    print(f"Data Processing Core: Warning - Some Chonkie chunkers may be unavailable: {e_chonk}")
    # Define dummy classes if specific Chonkie chunkers are critical for type hints
    class SDPMChunker:pass; 
    class SemanticChunker:pass; 
    class NeuralChunker:pass; 
    class RecursiveChunker:pass; 
    class SentenceChunker:pass; 
    class TokenChunker:pass;
    
from transformers import pipeline as hf_pipeline
try: import umap; UMAP_AVAILABLE = True
except ImportError: UMAP_AVAILABLE = False

WORKER_CHUNKER_INSTANCE: Optional[ChonkieChunkerInstanceType] = None 

def _init_worker_chunker(chunker_config: Dict[str, Any]):
    global WORKER_CHUNKER_INSTANCE
    chunker_type_name = chunker_config.get("type", "unknown")
    print(f"Worker Init: Attempting chunker type '{chunker_type_name}' with config keys: {list(chunker_config.keys())}")

    WORKER_CHUNKER_INSTANCE = None # Reset before trying
    if chunker_type_name == "semchunk":
        if not SEMCHUNK_AVAILABLE: print("Worker Error: Semchunk not available."); return
        WORKER_CHUNKER_INSTANCE = semchunk_main_chunkerify_func(
            tokenizer_or_token_counter=common_utils.count_tokens_robustly,
            chunk_size=chunker_config["semchunk_max_tokens_chunk"]
        )
    elif CHONKIE_AVAILABLE: # Common setup for most Chonkie chunkers
        chonkie_params = {}
        if chunker_type_name in ["chonkie_sdpm", "chonkie_semantic"]:
            chonkie_params["embedding_model"] = chunker_config["chonkie_embedding_model"]
            chonkie_params["chunk_size"] = chunker_config["chonkie_target_chunk_size"]
            chonkie_params["threshold"] = chunker_config["chonkie_similarity_threshold"]
            chonkie_params["min_sentences"] = chunker_config["chonkie_min_sentences"]
            chonkie_params["mode"] = chunker_config.get(f"{chunker_type_name}_mode", "window")
            chonkie_params["similarity_window"] = chunker_config.get(f"{chunker_type_name}_similarity_window", 1)
            chonkie_params["min_chunk_size"] = chunker_config.get(f"{chunker_type_name}_min_chunk_size", 20)
            chonkie_params["device"] = 'cpu' # Force CPU for Chonkie workers with embedding models
            if chunker_type_name == "chonkie_sdpm":
                chonkie_params["skip_window"] = chunker_config["chonkie_skip_window"]
            WORKER_CHUNKER_INSTANCE = SemanticChunker(**chonkie_params) if chunker_type_name == "chonkie_semantic" else SDPMChunker(**chonkie_params)
        elif chunker_type_name == "chonkie_neural":
            WORKER_CHUNKER_INSTANCE = NeuralChunker(
                model=chunker_config["chonkie_neural_model"],
                tokenizer=chunker_config.get("chonkie_neural_tokenizer", chunker_config["chonkie_neural_model"]), # Often tokenizer is same as model
                min_characters_per_chunk=chunker_config["chonkie_neural_min_chars"],
                stride=chunker_config["chonkie_neural_stride"],
                device_map='cpu' # For HF pipeline within NeuralChunker
            )
        elif chunker_type_name == "chonkie_recursive":
            WORKER_CHUNKER_INSTANCE = RecursiveChunker(
                tokenizer_or_token_counter=chunker_config.get("chonkie_recursive_tokenizer", common_utils.BGE_TOKENIZER_INSTANCE),
                chunk_size=chunker_config["chonkie_recursive_chunk_size"],
                min_characters_per_chunk=chunker_config["chonkie_recursive_min_chars"]
            )
        elif chunker_type_name == "chonkie_sentence":
            WORKER_CHUNKER_INSTANCE = SentenceChunker(
                tokenizer_or_token_counter=chunker_config.get("chonkie_sentence_tokenizer", common_utils.BGE_TOKENIZER_INSTANCE),
                chunk_size=chunker_config["chonkie_sentence_chunk_size"],
                chunk_overlap=chunker_config["chonkie_sentence_overlap"],
                min_sentences_per_chunk=chunker_config["chonkie_sentence_min_sents"]
            )
        elif chunker_type_name == "chonkie_token":
            WORKER_CHUNKER_INSTANCE = TokenChunker(
                tokenizer=chunker_config.get("chonkie_token_tokenizer", "gpt2"), # Chonkie's TokenChunker needs a tokenizer string/instance
                chunk_size=chunker_config["chonkie_token_chunk_size"],
                chunk_overlap=chunker_config["chonkie_token_overlap"]
            )
        else: print(f"Worker Init Error: Chonkie type '{chunker_type_name}' not handled or Chonkie unavailable.")
    else: # Chonkie not available and not semchunk
        print(f"Worker Init Error: No available chunker for type '{chunker_type_name}'.")
    
    if WORKER_CHUNKER_INSTANCE: print(f"Worker Chunker '{chunker_type_name}' initialized.")
    else: print(f"Worker Chunker '{chunker_type_name}' FAILED to initialize.")


def _extract_tabular_row_as_document(row: Dict[str, Any], column_names: List[str], base_metadata: dict, row_idx: int, sheet_name: Optional[str] = None) -> Optional[Tuple[str, dict]]:
    # Same as before
    text_parts = [f"{col}: {str(row.get(col))}" for col in column_names if row.get(col) is not None]
    content_str = " | ".join(text_parts); cleaned_content = common_utils.clean_text(content_str)
    if cleaned_content:
        doc_meta = {**base_metadata, "row_index": row_idx}; 
        if sheet_name: doc_meta["sheet_name"] = sheet_name
        for col, val in row.items():
            if val is not None: doc_meta[f"meta_{common_utils.sanitize_filename_for_id(str(col))}"] = val
        return cleaned_content, doc_meta
    return None

def extract_text_from_json_value(value, current_path="root", path_sep="."): # Same
    texts_with_paths = [];
    if isinstance(value, dict):
        if not value: return []
        for k, v_item in value.items(): new_path = f"{current_path}{path_sep}{k}" if current_path!="root" else k; texts_with_paths.extend(extract_text_from_json_value(v_item, new_path, path_sep))
    elif isinstance(value, list):
        if not value: return []
        for i, item_val in enumerate(value): new_path = f"{current_path}{path_sep}[{i}]" if current_path!="root" else f"[{i}]"; texts_with_paths.extend(extract_text_from_json_value(item_val, new_path, path_sep))
    elif isinstance(value, (str, int, float, bool)):
        cleaned = common_utils.clean_text(str(value));
        if cleaned: texts_with_paths.append({"text": cleaned, "json_path": current_path})
    return texts_with_paths

def extract_documents_from_file(filepath: str, relative_path: str) -> list[tuple[str, dict]]: # Same as before, ensures robustness
    # (File extraction logic for all types remains the same as the previous fully elaborated one)
    docs_with_metadata = []; base_metadata = {"source_file": relative_path, "file_type": os.path.splitext(filepath)[1].lower(), "file_last_modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat()}; file_type = base_metadata["file_type"]
    try:
        content_str: str = ""
        if file_type in [".txt", ".md", ".py", ".js", ".html", ".css", ".xml", ".sh", ".bat", ".yaml", ".yml", ".ini", ".cfg", ".toml"]:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f: content_str = f.read(); cleaned = common_utils.clean_text(content_str);
            if cleaned: docs_with_metadata.append((cleaned, base_metadata.copy()))
        elif file_type == ".csv":
            df = pl.read_csv(filepath, infer_schema_length=100, truncate_ragged_lines=True, ignore_errors=True, encoding="utf-8-lossy"); column_names = df.columns
            for i, row_dict in enumerate(df.iter_rows(named=True)):
                doc_tuple = _extract_tabular_row_as_document(row_dict, column_names, base_metadata, i);
                if doc_tuple: docs_with_metadata.append(doc_tuple)
        elif file_type == ".parquet": 
            df = pl.read_parquet(filepath); column_names = df.columns
            for i, row_dict in enumerate(df.iter_rows(named=True)):
                doc_tuple = _extract_tabular_row_as_document(row_dict, column_names, base_metadata, i);
                if doc_tuple: docs_with_metadata.append(doc_tuple)
        elif file_type == ".xlsx":
            try:
                all_sheets = pl.read_excel(filepath, sheet_name=None, engine='openpyxl', read_options={"infer_schema_length": 100})
                for sheet_name, df_sheet in all_sheets.items():
                    if df_sheet.height == 0: continue; sheet_base_meta = {**base_metadata, "sheet_name": common_utils.sanitize_filename_for_id(sheet_name)}; column_names = df_sheet.columns
                    for i, row_dict in enumerate(df_sheet.iter_rows(named=True)):
                        doc_tuple = _extract_tabular_row_as_document(row_dict, column_names, sheet_base_meta, i);
                        if doc_tuple: docs_with_metadata.append(doc_tuple)
            except Exception as e_xlsx: print(f"  Warn: Polars Excel read fail for {filepath}, trying openpyxl direct: {e_xlsx}") # Keep fallback
        elif file_type == ".pdf":
            try:
                reader = PdfReader(filepath)
                if not reader.pages: return [];
                for i, page in enumerate(reader.pages): cleaned = common_utils.clean_text(page.extract_text() or "");
                if cleaned: docs_with_metadata.append((cleaned, {**base_metadata, "page_number": i + 1}))
            except Exception as e_pdf: print(f" Error processing PDF {filepath}: {e_pdf}")
        elif file_type == ".docx":
            try:
                doc = DocxDocument(filepath)
                cleaned = common_utils.clean_text(
                    "\n".join([p.text for p in doc.paragraphs if p.text])
                )
                if cleaned: docs_with_metadata.append((cleaned, base_metadata.copy()))
            except Exception as e_docx: print(f" Error processing DOCX {filepath}: {e_docx}")
        elif file_type == ".json":
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f: data = json.load(f)
            for item in extract_text_from_json_value(data): docs_with_metadata.append((item["text"], {**base_metadata, "json_path": item["json_path"]}))
        elif file_type == ".jsonl":
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    try: data = json.loads(line)
                    except json.JSONDecodeError: print(f" Skip invalid JSONL line {i+1} in {filepath}"); continue
                    for item in extract_text_from_json_value(data): docs_with_metadata.append((item["text"], {**base_metadata, "line_number": i + 1, "json_path": item["json_path"]}))
        elif file_type == ".epub":
            try:
                book = epub.read_epub(filepath)
                for item in book.get_items_of_type(ITEM_DOCUMENT):
                    cleaned = common_utils.clean_text(BeautifulSoup(item.get_content(), 'html.parser').get_text(separator='\n'))
                    if cleaned:
                        meta = {
                            **base_metadata,
                            "epub_item_id": item.get_id(),
                            "epub_item_name": item.get_name(),
                        }
                    try:
                        title_meta = book.get_metadata("DC", "title")
                    except pass
                    else: 
                        if title_meta: meta["epub_book_title"] = title_meta[0][0]
                    docs_with_metadata.append((cleaned, meta))
            except Exception as e_epub: print(f" Error processing EPUB {filepath}: {e_epub}")
        return docs_with_metadata
    except Exception as e_main_extract: print(f"  Major error extracting from {filepath}: {e_main_extract}"); return []


def _chunk_doc_content_worker_entry(args_tuple: Tuple) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    # Same as before, relies on WORKER_CHUNKER_INSTANCE
    doc_content, base_metadata, doc_id_prefix, chunker_params = args_tuple
    if WORKER_CHUNKER_INSTANCE is None:
        # Determine a fallback chunk size from params, or a hardcoded default
        fallback_max_len = 500 * 5 # default character limit
        if chunker_params:
            if "semchunk_max_tokens_chunk" in chunker_params: fallback_max_len = chunker_params["semchunk_max_tokens_chunk"] * 5
            elif "chonkie_target_chunk_size" in chunker_params: fallback_max_len = chunker_params["chonkie_target_chunk_size"] * 5
            elif "chonkie_recursive_chunk_size" in chunker_params: fallback_max_len = chunker_params["chonkie_recursive_chunk_size"] * 5
        
        chunk_text = doc_content[:fallback_max_len]
        vic_id = f"{doc_id_prefix}_chunk_0_fallback"; disp_name = generate_display_source_name(base_metadata,0)
        meta = {**base_metadata, "chunk_id_within_doc":0,"vicinity_item_id":vic_id,"display_source_name":disp_name,"chunking_error":"Worker chunker not initialized/available"}
        return [chunk_text] if chunk_text.strip() else [], [vic_id] if chunk_text.strip() else [], [meta] if chunk_text.strip() else []

    text_chunks_for_doc, chunk_ids_for_doc, detailed_meta_for_doc = [], [], []
    chunk_texts_list: List[str] = []
    chunker_type_name = chunker_params["type"]
    try:
        if chunker_type_name == "semchunk":
            overlap = int(chunker_params["semchunk_max_tokens_chunk"] * (chunker_params["semchunk_overlap_percent"] / 100.0))
            paras = [p.strip() for p in doc_content.split('\n\n') if p.strip()] or [doc_content]
            list_of_lists_of_strings = WORKER_CHUNKER_INSTANCE(paras, overlap=overlap)
            chunk_texts_list = [chunk for sublist in list_of_lists_of_strings for chunk in sublist if chunk.strip()]
        elif chunker_type_name.startswith("chonkie_"):
            # Chonkie chunkers have a .chunk() method that returns a list of Chonkie Chunk objects or texts
            chonkie_output_objects = WORKER_CHUNKER_INSTANCE.chunk(doc_content) 
            if chonkie_output_objects and isinstance(chonkie_output_objects[0], str): # if return_type='texts'
                chunk_texts_list = [text for text in chonkie_output_objects if text.strip()]
            else: # if return_type='chunks' (default for most Chonkie chunkers)
                chunk_texts_list = [getattr(c, 'text', str(c)) for c in chonkie_output_objects if hasattr(c, 'text') and getattr(c, 'text').strip()]
        else: chunk_texts_list = [doc_content]; base_metadata["chunking_error"] = f"Unknown chunker type {chunker_type_name}"
    except Exception as e_chunk:
        print(f"Worker Error chunking {doc_id_prefix} with {chunker_type_name}: {e_chunk}")
        fallback_max_len = 500 * 5 # Default char limit if specific chunker params not found
        if chunker_params:
             if "semchunk_max_tokens_chunk" in chunker_params: fallback_max_len = chunker_params["semchunk_max_tokens_chunk"] * 5
             elif "chonkie_target_chunk_size" in chunker_params: fallback_max_len = chunker_params["chonkie_target_chunk_size"] * 5
        chunk_texts_list = [doc_content[:fallback_max_len]]; base_metadata["chunking_error"] = str(e_chunk)

    for idx, text_item in enumerate(chunk_texts_list):
        if text_item.strip():
            vic_id = f"{doc_id_prefix}_chunk_{idx}"; text_chunks_for_doc.append(text_item); chunk_ids_for_doc.append(vic_id)
            meta = {**base_metadata, "chunk_id_within_doc":idx, "vicinity_item_id":vic_id, "display_source_name": generate_display_source_name(base_metadata,idx)}
            detailed_meta_for_doc.append(meta)       
    if not text_chunks_for_doc and doc_content.strip():
        fallback_max_len = 500*5 # As above
        chunk_text_fb = doc_content[:fallback_max_len]
        vic_id = f"{doc_id_prefix}_chunk_0_full"; text_chunks_for_doc.append(chunk_text_fb); chunk_ids_for_doc.append(vic_id)
        meta_fb = {**base_metadata, "chunk_id_within_doc":0, "vicinity_item_id": vic_id, "display_source_name": generate_display_source_name(base_metadata,0)}
        detailed_meta_for_doc.append(meta_fb)
    return text_chunks_for_doc, chunk_ids_for_doc, detailed_meta_for_doc

def initialize_classifier(model_name: str, device_id: int = -1): # Same
    try: 
        effective_device = device_id if torch.cuda.is_available() and device_id >=0 else -1
        print(f"Data Proc Core: Init Classifier '{model_name}' (TargetDev: {device_id}, EffDev: {'cpu' if effective_device == -1 else f'cuda:{effective_device}'})...");
        classifier = hf_pipeline("zero-shot-classification", model=model_name, device=effective_device);
        print(f"Data Proc Core: Classifier loaded on device: {classifier.device}.")
        return classifier
    except Exception as e_clf: print(f"Data Proc Core: Error init classifier {model_name}: {e_clf}. Classification skipped."); return None

def classify_chunks_batch(chunks: list[str], classifier, labels: list[str], batch_size: int, desc: str = "Classifying Batches") -> list[dict | None]: # Same
    if classifier is None or not chunks: return [{"classification_status": "skipped_no_classifier_or_chunks"}] * len(chunks)
    results = []
    try:
        for i in tqdm(range(0, len(chunks), batch_size), desc=desc):
            batch = chunks[i:i + batch_size];
            if not batch: continue
            outputs = classifier(batch, candidate_labels=labels, multi_label=False)
            if isinstance(outputs, dict) and len(batch) == 1: outputs = [outputs]
            elif not isinstance(outputs, list): results.extend([{"classification_error": "unexpected_output"}] * len(batch)); continue
            for res_item in outputs: 
                results.append({"top_label": res_item["labels"][0] if res_item.get("labels") else "N/A", 
                                "top_label_score": float(res_item["scores"][0]) if res_item.get("scores") else 0.0, 
                                "zero_shot_labels": res_item.get("labels", []), 
                                "zero_shot_scores": [float(s) for s in res_item.get("scores", [])]})
        return results
    except Exception as e_clf_batch: print(f"Error during batch classification: {e_clf_batch}"); return [{"classification_error": str(e_clf_batch)}] * len(chunks)


def prepare_documents_and_chunks(
    docs_input_folder: str, 
    # Config dictionary will hold all chunker-specific params
    chunker_config: Dict[str, Any], 
    num_workers_cfg: int, enable_clf_cfg: bool, clf_model_name_cfg: Optional[str], 
    clf_labels_cfg: Optional[List[str]], clf_batch_size_cfg: int, 
    gpu_id_cfg: int, verbose_logging_cfg: bool
) -> tuple[list, list, list]:
    
    extracted_doc_parts = [] 
    file_paths = [os.path.join(r,f) for r,_,fs in os.walk(docs_input_folder) for f in fs if not f.startswith(('.', '__MACOSX'))]
    if not file_paths: print("No processable files found."); return [], [], []
    print(f"PHASE 1: Text Extraction - Found {len(file_paths)} files.")

    for fpath in tqdm(file_paths, desc="Extracting File Contents", disable=not verbose_logging_cfg):
        relative_fpath = os.path.relpath(fpath, docs_input_folder); sanitized_base_id = common_utils.sanitize_filename_for_id(relative_fpath)
        docs_from_file = extract_documents_from_file(fpath, relative_fpath)
        for doc_idx, (content, meta) in enumerate(docs_from_file):
            id_prefix = sanitized_base_id; id_suffix_parts = []
            for key in ["page_number", "sheet_name", "row_index", "line_number", "json_path", "epub_item_id"]:
                if key in meta: id_suffix_parts.append(f"{key.split('_')[0][:1]}{common_utils.sanitize_filename_for_id(str(meta[key]))[:15]}")
            current_id_prefix = f"{sanitized_base_id}_{'_'.join(id_suffix_parts)}" if id_suffix_parts else (f"{sanitized_base_id}_docpart{doc_idx}" if len(docs_from_file)>1 else sanitized_base_id)
            current_id_prefix = common_utils.sanitize_filename_for_id(current_id_prefix)
            if content and content.strip(): extracted_doc_parts.append((content, meta, current_id_prefix))
    if not extracted_doc_parts: print("No text content extracted."); return [], [], []

    print(f"\nPHASE 2: Chunking {len(extracted_doc_parts)} doc parts (Workers: {num_workers_cfg}, Chunker: {chunker_config.get('type','N/A').upper()})...")
    tasks_for_pool = [(part[0], part[1], part[2], chunker_config) for part in extracted_doc_parts] # Pass the whole config
    all_text_chunks, all_chunk_ids, all_detailed_meta = [], [], []
    
    pool_initializer_func = _init_worker_chunker; pool_initargs_tuple = (chunker_config,)
    if num_workers_cfg > 1 and len(tasks_for_pool) > 1:
        with multiprocessing.Pool(processes=num_workers_cfg, initializer=pool_initializer_func, initargs=pool_initargs_tuple) as pool:
            results = list(tqdm(pool.imap_unordered(_chunk_doc_content_worker_entry, tasks_for_pool), total=len(tasks_for_pool), desc="Parallel Chunking"))
    else: 
        pool_initializer_func(*pool_initargs_tuple)
        results = [_chunk_doc_content_worker_entry(task_args) for task_args in tqdm(tasks_for_pool, desc="Sequential Chunking")]
    for tc_list, cid_list, dm_list in results:
        all_text_chunks.extend(tc_list); all_chunk_ids.extend(cid_list); all_detailed_meta.extend(dm_list)
    if not all_text_chunks: print("No chunks generated."); return [], [], []
    print(f"Total chunks: {len(all_text_chunks)}")

    if enable_clf_cfg and clf_model_name_cfg and all_text_chunks:
        print(f"\nPHASE 3: Classifying {len(all_text_chunks)} chunks...")
        clf_instance = initialize_classifier(clf_model_name_cfg, device_id=gpu_id_cfg)
        if clf_instance:
            class_res = classify_chunks_batch(all_text_chunks, clf_instance, clf_labels_cfg or [], clf_batch_size_cfg, "Classifying Chunks")
            if len(class_res) == len(all_detailed_meta): [all_detailed_meta[i].update(res) for i, res in enumerate(class_res) if res]
            else: print(f"Warning: Classification results len != metadata len.")
    return all_text_chunks, all_chunk_ids, all_detailed_meta

def save_data_for_visualization( # Same
    vector_data_dir: str, chunk_vectors: np.ndarray, metadata_list: list[dict], chunk_texts_list: list[str],
    output_viz_filename: str, umap_n_neighbors: int, umap_min_dist_val: float, umap_metric_val: str):
    if not UMAP_AVAILABLE: print("UMAP lib not found. Skipping viz data save."); return
    if not (len(chunk_vectors) == len(metadata_list) == len(chunk_texts_list)): print("Error: Mismatch for viz data."); return
    if len(chunk_vectors) == 0: print("No vector data for viz."); return
    print(f"\nGenerating UMAP for visualization ({len(chunk_vectors)} points)...")
    try:
        reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=umap_min_dist_val, n_components=2, metric=umap_metric_val, random_state=42, verbose=False) 
        embedding_2d = reducer.fit_transform(chunk_vectors)
        viz_data_points = [{'id': m.get('vicinity_item_id',f'chunk_{i}'), 'x': float(embedding_2d[i,0]), 'y': float(embedding_2d[i,1]),
                 'source_file': m.get('source_file','N/A'), 'display_source': m.get('display_source_name','N/A'),
                 'classification': m.get('top_label','N/A'), 'snippet': c[:200]+"..." } for i, (m,c) in enumerate(zip(metadata_list, chunk_texts_list))]
        output_path = os.path.join(vector_data_dir, output_viz_filename)
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(viz_data_points, f)
        print(f"Visualization data saved to: {output_path}")
    except Exception as e_umap: print(f"Error during UMAP/saving viz data: {e_umap}")