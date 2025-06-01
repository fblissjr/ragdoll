# src/ragdoll_utils.py
# This module provides common utility functions used across the RAGdoll project.
# It includes text cleaning, ID sanitization, token counting, and display name generation.

import os
import re
from typing import Dict, Any # For type hinting

# Attempt to import transformers and set up tokenizer
# This is a core utility, so its initialization happens on module import.
try:
    from transformers import AutoTokenizer as HFAutoTokenizer
    # Load the BGE tokenizer, widely used for RAG tasks due to its performance.
    BGE_TOKENIZER_INSTANCE = HFAutoTokenizer.from_pretrained("baai/bge-base-en-v1.5")
    TOKEN_COUNT_FALLBACK_ACTIVE = False
    print("Ragdoll Utils: BGE Tokenizer (baai/bge-base-en-v1.5) loaded successfully.")
except Exception as e_bge_load:
    print(f"Ragdoll Utils: WARNING - Error loading BGE tokenizer: {e_bge_load}. Using basic split() for token counting.")
    BGE_TOKENIZER_INSTANCE = None # Explicitly set to None on failure
    TOKEN_COUNT_FALLBACK_ACTIVE = True

def count_tokens_robustly(text_to_count: str) -> int:
    """
    Counts tokens in a given text. Uses the BGE tokenizer if available, 
    otherwise falls back to a simple word split count.
    This is crucial for estimating context window sizes for LLMs.
    """
    text_str = str(text_to_count)
    if TOKEN_COUNT_FALLBACK_ACTIVE or BGE_TOKENIZER_INSTANCE is None:
        # Fallback to word count if BGE tokenizer failed to load or is None
        return len(text_str.split())
    
    # BGE_TOKENIZER_INSTANCE is an HF AutoTokenizer instance.
    # `encode` method tokenizes the text and returns a list of token IDs.
    return len(BGE_TOKENIZER_INSTANCE.encode(text_str, add_special_tokens=False))


def clean_text(text: str) -> str:
    """
    Cleans a text string by consolidating whitespace and removing null bytes.
    Essential for preprocessing text before chunking or embedding.
    """
    text_str = str(text) if text is not None else ""
    text_str = re.sub(r'\s+', ' ', text_str) # Consolidate multiple whitespace characters into a single space
    text_str = re.sub(r'\x00', '', text_str)  # Remove NULL bytes, which can cause issues in processing
    return text_str.strip() # Remove leading/trailing whitespace

def sanitize_filename_for_id(filename: str) -> str:
    """
    Sanitizes a filename to be used as a component in an ID.
    Replaces path separators and invalid characters with underscores.
    Ensures IDs are clean and can be used in file systems or as database keys.
    """
    s = str(filename).replace(os.path.sep, "_") # Replace OS-specific path separators with underscores
    s = re.sub(r'[^0-9a-zA-Z_.-]', '_', s) # Allow only alphanumeric, underscore, dot, hyphen; replace others with underscore
    s = re.sub(r'_+', '_', s) # Consolidate multiple underscores
    s = s.strip('_-.') # Remove leading/trailing underscores, dots, or hyphens
    return s if s else "unknown_file_component" # Return a default if the string becomes empty

def generate_display_source_name(metadata: Dict[str, Any], chunk_order_in_doc_part: int) -> str:
    """
    Generates a human-readable display name for a chunk based on its metadata.
    This helps users identify the origin of a chunk in UI or logs.
    chunk_order_in_doc_part is assumed to be 0-indexed from processing.
    """
    source_file = metadata.get("source_file", "UnknownSource")
    # Use the base name of the file to keep it concise
    base_name = os.path.basename(source_file) 
    
    parts = [base_name] # Start with the filename
    
    # Add specific location identifiers from metadata if available
    if "sheet_name" in metadata:
        parts.append(f"Sht:{metadata['sheet_name']}")
    if "page_number" in metadata:
        parts.append(f"Pg:{metadata['page_number']}")
    if "row_index" in metadata: # Assuming 0-indexed from internal processing
        parts.append(f"Rw:{metadata['row_index'] + 1}") 
    if "line_number" in metadata: # For JSONL or similar line-based formats
        parts.append(f"Ln:{metadata['line_number']}")
    if "json_path" in metadata:
        jp = metadata['json_path']
        # Shorten very long JSON paths for display
        jp_display = jp if len(jp) < 25 else f"...{jp[-22:]}" 
        parts.append(f"Path:{jp_display}")
    if "epub_item_name" in metadata: # For EPUB chapters/sections
         parts.append(f"Item:{metadata['epub_item_name']}")
    
    # Add the chunk index (1-based for display)
    # 'chunk_order_in_doc_part' is preferred if available, otherwise use general 'chunk_index'
    actual_chunk_idx = metadata.get('chunk_order_in_doc_part', chunk_order_in_doc_part)
    parts.append(f"Chk:{actual_chunk_idx + 1}") 

    return " | ".join(parts) # Join all parts with a separator

# Final check for BGE Tokenizer status after all imports within this module (if any affected it)
if BGE_TOKENIZER_INSTANCE is None and not TOKEN_COUNT_FALLBACK_ACTIVE:
    # This should ideally not happen if the initial try-except is comprehensive
    print("Ragdoll Utils: Critical - BGE Tokenizer instance is None but fallback was not initially active. Activating fallback for safety.")
    TOKEN_COUNT_FALLBACK_ACTIVE = True