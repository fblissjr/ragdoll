# src/ragdoll_utils.py
# This module provides common utility functions used across the RAGdoll project.
# It includes text cleaning, ID sanitization, token counting, and display name generation.

import os
import re
from typing import Dict, Any, Optional

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

def generate_display_source_name(metadata: Dict[str, Any], chunk_index: Optional[int] = None) -> str:
    """
    Generates a user-friendly display name for a chunk based on its metadata.
    Example: "MyDocument.pdf (Page 3, Chunk 2)"
    """
    source_file = metadata.get("source_file", "UnknownSource")
    base_name = os.path.basename(source_file)
    
    name_parts = [base_name]
    
    # Add page number if available
    if "page_number" in metadata and metadata["page_number"] is not None:
        name_parts.append(f"P:{metadata['page_number']}")
    
    # Add sheet name if available (for Excel files)
    if "sheet_name" in metadata and metadata["sheet_name"] is not None:
        name_parts.append(f"Sht:{metadata['sheet_name'][:15]}") # Truncate long sheet names
        
    # Add row index if available (for tabular data)
    if "row_index" in metadata and metadata["row_index"] is not None:
        name_parts.append(f"Row:{metadata['row_index'] + 1}") # 1-based for display

    # Add line number if available (for JSONL or line-based text)
    if "line_number" in metadata and metadata["line_number"] is not None:
        name_parts.append(f"L:{metadata['line_number']}")
        
    # Add JSON path if available
    if "json_path" in metadata and metadata["json_path"] is not None:
        path_suffix = metadata["json_path"]
        if len(path_suffix) > 20: # Truncate long JSON paths
            path_suffix = "..." + path_suffix[-17:]
        name_parts.append(f"Path:{path_suffix}")

    # Add EPUB item name if available
    if "epub_item_name" in metadata and metadata["epub_item_name"] is not None:
        epub_name_suffix = metadata["epub_item_name"]
        if len(epub_name_suffix) > 20:
            epub_name_suffix = epub_name_suffix[:17] + "..."
        name_parts.append(f"Epub:{epub_name_suffix}")
        
    # Add the chunk index within its document part if provided
    if chunk_index is not None:
        name_parts.append(f"Chk:{chunk_index + 1}") # Display as 1-based index

    if len(name_parts) > 1:
        return f"{name_parts[0]} ({', '.join(name_parts[1:])})"
    else:
        return name_parts[0]

# Final check for BGE Tokenizer status after all imports within this module (if any affected it)
if BGE_TOKENIZER_INSTANCE is None and not TOKEN_COUNT_FALLBACK_ACTIVE:
    # This should ideally not happen if the initial try-except is comprehensive
    print("Ragdoll Utils: Critical - BGE Tokenizer instance is None but fallback was not initially active. Activating fallback for safety.")
    TOKEN_COUNT_FALLBACK_ACTIVE = True