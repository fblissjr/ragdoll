# src/core_logic/file_parser.py
# This module is responsible for extracting text content and metadata from various file formats.

import os
import json
import re
import polars as pl
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Third-party libraries for specific file types
from pypdf import PdfReader
import pypandoc # Requires Pandoc to be installed on the system
from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

# Project-specific utilities
from .. import ragdoll_utils # Assuming utils is one level up if core_logic is a subfolder of src

# --- Helper Functions for Specific Extraction Tasks ---

def _extract_tabular_row_as_document(row: Dict[str, Any], column_names: List[str], base_metadata: dict, row_idx: int, sheet_name: Optional[str] = None) -> Optional[Tuple[str, dict]]:
    """
    Converts a single row from a tabular data source into a string document ("key: value" pairs)
    and associated metadata.
    """
    text_parts = [f"{col}: {str(row.get(col))}" for col in column_names if row.get(col) is not None and str(row.get(col)).strip()]
    if not text_parts: return None # Skip if row is effectively empty after stripping values

    content_str = " | ".join(text_parts)
    cleaned_content = ragdoll_utils.clean_text(content_str)

    if cleaned_content:
        doc_meta = {**base_metadata, "row_index": row_idx} # row_idx is 0-based from iter_rows
        if sheet_name: 
            doc_meta["sheet_name"] = sheet_name
        for col, val in row.items():
            if val is not None: 
                doc_meta[f"meta_{ragdoll_utils.sanitize_filename_for_id(str(col))}"] = str(val) # Store original values as metadata
        return cleaned_content, doc_meta
    return None

def _extract_text_from_json_value_recursive(value: Any, current_path: str = "root", path_sep: str = ".") -> List[Dict[str,str]]:
    """
    Recursively extracts all string-like values from a JSON structure (dict or list).
    Returns a list of dictionaries, each containing the extracted 'text' and its 'json_path'.
    """
    texts_with_paths = []
    if isinstance(value, dict):
        if not value: return []
        for k, v_item in value.items(): 
            new_path = f"{current_path}{path_sep}{k}" if current_path != "root" else k
            texts_with_paths.extend(_extract_text_from_json_value_recursive(v_item, new_path, path_sep))
    elif isinstance(value, list):
        if not value: return []
        for i, item_val in enumerate(value): 
            new_path = f"{current_path}{path_sep}[{i}]" if current_path != "root" else f"[{i}]"
            texts_with_paths.extend(_extract_text_from_json_value_recursive(item_val, new_path, path_sep))
    elif isinstance(value, (str, int, float, bool)): # Convert primitives to string
        cleaned = ragdoll_utils.clean_text(str(value))
        if cleaned: 
            texts_with_paths.append({"text": cleaned, "json_path": current_path})
    return texts_with_paths

# --- Main Extraction Functions per File Category ---

def _extract_plain_text_like(filepath: str, base_metadata: dict) -> list[tuple[str, dict]]:
    """
    Handles plain text files and common text-based formats like Markdown, code files, YAML, etc.
    Reads the entire file as one document part.
    """
    docs_with_metadata = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f: 
            content_str = f.read()
        cleaned_content = ragdoll_utils.clean_text(content_str)
        if cleaned_content: 
            meta = {**base_metadata, "content_type": "text"} 
            # Specifically mark Markdown content for potential specialized chunking
            if base_metadata.get("file_type") == ".md": 
                meta["content_type"] = "markdown"
            docs_with_metadata.append((cleaned_content, meta))
    except Exception as e:
        print(f"  File Parser Error: Processing plain text-like file '{filepath}': {e}")
    return docs_with_metadata

def _extract_tabular_formats(filepath: str, base_metadata: dict, file_type: str) -> list[tuple[str, dict]]:
    """
    Extracts data from tabular files (.csv, .parquet, .xlsx). Each row is treated as a document.
    Uses Polars for efficient reading.
    """
    docs_with_metadata = []
    try:
        df_list = [] # To handle multiple dataframes (e.g., sheets in Excel)
        common_content_type = "table"

        if file_type == ".csv":
            df_list.append(pl.read_csv(filepath, infer_schema_length=1000, truncate_ragged_lines=True, ignore_errors=True, encoding="utf-8-lossy"))
        elif file_type == ".parquet":
            df_list.append(pl.read_parquet(filepath))
        elif file_type == ".xlsx":
            common_content_type = "table_sheet" # More specific for Excel sheets
            all_sheets_dict = pl.read_excel(filepath, sheet_name=None, engine='openpyxl', read_options={"infer_schema_length": 1000})
            for sheet_name_raw, df_sheet in all_sheets_dict.items():
                if df_sheet.height > 0: # Process non-empty sheets
                    sane_sheet_name = ragdoll_utils.sanitize_filename_for_id(str(sheet_name_raw))
                    # Pass sheet_name to be included in metadata for each row from this sheet
                    df_list.append((df_sheet, sane_sheet_name)) 
                else:
                    print(f"  File Parser Info: Skipping empty sheet '{sheet_name_raw}' in '{filepath}'.")
        
        # Process each dataframe (or sheet-dataframe tuple for Excel)
        for item in df_list:
            current_df = None
            sheet_specific_name = None
            if file_type == ".xlsx": # Excel df_list contains (df, sheet_name) tuples
                current_df, sheet_specific_name = item
            else: # CSV/Parquet df_list contains just the dataframe
                current_df = item

            if current_df is not None and current_df.height > 0:
                column_names = current_df.columns
                # Prepare metadata for this table/sheet
                table_meta = {**base_metadata, "content_type": common_content_type}
                if sheet_specific_name:
                    table_meta["sheet_name"] = sheet_specific_name # Add sheet name if applicable

                for i, row_dict in enumerate(current_df.iter_rows(named=True)):
                    doc_tuple = _extract_tabular_row_as_document(row_dict, column_names, table_meta, i, sheet_name=sheet_specific_name)
                    if doc_tuple: 
                        docs_with_metadata.append(doc_tuple)
            elif current_df is not None and current_df.height == 0:
                 print(f"  File Parser Info: Dataframe for '{filepath}' (sheet: {sheet_specific_name or 'N/A'}) is empty.")

    except Exception as e:
        print(f"  File Parser Error: Processing tabular file '{filepath}' (type {file_type}): {e}")
    return docs_with_metadata

def _extract_pdf_format(filepath: str, base_metadata: dict) -> list[tuple[str, dict]]:
    """
    Extracts text from each page of a PDF file. Each page becomes a document part.
    """
    docs_with_metadata = []
    try:
        reader = PdfReader(filepath)
        if not reader.pages: 
            print(f"  File Parser Warn: PDF '{filepath}' has no pages or is unreadable by pypdf.")
            return []
        for i, page in enumerate(reader.pages): 
            page_text = page.extract_text() or "" 
            cleaned_content = ragdoll_utils.clean_text(page_text)
            if cleaned_content: 
                meta = {**base_metadata, "page_number": i + 1, "content_type": "pdf_page"}
                docs_with_metadata.append((cleaned_content, meta))
    except Exception as e: 
        print(f"  File Parser Error: Processing PDF file '{filepath}': {e}")
    return docs_with_metadata

def _extract_docx_to_markdown(filepath: str, base_metadata: dict) -> list[tuple[str, dict]]:
    """
    Converts DOCX files to Markdown using Pandoc. The entire Markdown content becomes one document part.
    Falls back to basic text extraction using python-docx if Pandoc is not found or fails.
    """
    docs_with_metadata = []
    try:
        pandoc_path = pypandoc.get_pandoc_path() # Raises OSError if Pandoc is not found
        
        # Define a folder for Pandoc to extract media (images). This is good practice.
        # The folder name includes sanitized filename to avoid conflicts if processing multiple DOCX.
        media_output_folder = os.path.join(
            os.path.dirname(filepath), # Store media in the same directory as the source DOCX
            "pandoc_media_" + ragdoll_utils.sanitize_filename_for_id(os.path.basename(filepath))
        )
        # os.makedirs(media_output_folder, exist_ok=True) # Create if it doesn't exist, if needed

        markdown_content = pypandoc.convert_file(
            filepath, 
            'gfm+hard_line_breaks', # Target GitHub Flavored Markdown, preserving intentional line breaks
            extra_args=['--wrap=none', f'--extract-media={media_output_folder}']
        )
        cleaned_markdown = ragdoll_utils.clean_text(markdown_content)
        if cleaned_markdown:
            meta = {**base_metadata, "content_type": "markdown", "extraction_method": "pandoc"}
            docs_with_metadata.append((cleaned_markdown, meta))
        else:
            print(f"  File Parser Warn: Pandoc conversion of DOCX '{filepath}' resulted in empty content.")

    except OSError: 
        print(f"  File Parser Error: Pandoc not found. DOCX '{filepath}' cannot be converted to Markdown. "
              "Ensure Pandoc is installed and accessible in the system PATH.")
        # Fallback to python-docx for basic text extraction
        try:
            from docx import Document as DocxDocument # Local import for fallback only
            print(f"  Attempting fallback DOCX text extraction for '{filepath}' using python-docx...")
            doc = DocxDocument(filepath)
            fallback_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            cleaned_fallback_text = ragdoll_utils.clean_text(fallback_text)
            if cleaned_fallback_text: 
                meta = {**base_metadata, "content_type": "text", "extraction_method": "python-docx-fallback", "pandoc_error": "pandoc_not_found"}
                docs_with_metadata.append((cleaned_fallback_text, meta))
        except Exception as e_docx_fallback:
            print(f"  File Parser Error: python-docx fallback failed for '{filepath}': {e_docx_fallback}")
            
    except Exception as e_docx_pandoc: 
        print(f"  File Parser Error: Pandoc conversion for DOCX '{filepath}' failed: {e_docx_pandoc}.")
        # Consider adding the python-docx fallback here as well for other Pandoc errors.
    return docs_with_metadata

def _extract_json_formats(filepath: str, base_metadata: dict, file_type: str) -> list[tuple[str, dict]]:
    """
    Extracts text from JSON (.json) and JSON Lines (.jsonl) files.
    Recursively traverses the JSON structure to find all string-like values.
    """
    docs_with_metadata = []
    try:
        if file_type == ".json":
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f: 
                data = json.load(f)
            for item in _extract_text_from_json_value_recursive(data): # Use the renamed recursive helper
                meta = {**base_metadata, "json_path": item["json_path"], "content_type": "json_value"}
                docs_with_metadata.append((item["text"], meta))
        elif file_type == ".jsonl":
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                for i, line in enumerate(f):
                    line_content = line.strip()
                    if not line_content: continue # Skip empty lines
                    try: 
                        data = json.loads(line_content)
                    except json.JSONDecodeError: 
                        print(f"  File Parser Warn: Skipping invalid JSONL line {i+1} in '{filepath}'."); continue
                    for item in _extract_text_from_json_value_recursive(data): 
                        meta = {**base_metadata, "line_number": i + 1, "json_path": item["json_path"], "content_type": "jsonl_value"}
                        docs_with_metadata.append((item["text"], meta))
    except Exception as e:
        print(f"  File Parser Error: Processing JSON file '{filepath}' (type {file_type}): {e}")
    return docs_with_metadata

def _extract_epub_format(filepath: str, base_metadata: dict) -> list[tuple[str, dict]]:
    """
    Extracts content from EPUB files. Each XHTML document item (chapter/section)
    within the EPUB is treated as a separate document part.
    """
    docs_with_metadata = []
    try:
        book = epub.read_epub(filepath)
        for item in book.get_items_of_type(ITEM_DOCUMENT): # ITEM_DOCUMENT usually refers to XHTML content
            content_bytes = item.get_content() # Raw content, typically XHTML bytes
            # Parse the XHTML content using BeautifulSoup
            soup = BeautifulSoup(content_bytes, 'html.parser') 
            
            # TODO: Future enhancement - convert XHTML from soup to Markdown using a library like 'markdownify'
            # This would provide more structured text for chunkers like RecursiveChunker (Markdown recipe).
            # For now, extracting plain text.
            raw_text_content = soup.get_text(separator='\n') # Use newline as separator for readability
            cleaned_content = ragdoll_utils.clean_text(raw_text_content)
            current_content_type = "epub_xhtml_content" # Reflects that it's text from XHTML
            
            if cleaned_content:
                meta = {
                    **base_metadata,
                    "epub_item_id": item.get_id(),       # Internal ID of the item in the EPUB package
                    "epub_item_name": item.get_name(),   # Filename or name of the item in EPUB manifest
                    "content_type": current_content_type
                }
                try: # Attempt to get the book title from EPUB's Dublin Core metadata
                    title_meta_dc = book.get_metadata("DC", "title")
                    if title_meta_dc and title_meta_dc[0] and title_meta_dc[0][0]: 
                        meta["epub_book_title"] = title_meta_dc[0][0]
                except: pass # Ignore if title metadata is not found or malformed
                docs_with_metadata.append((cleaned_content, meta))
    except Exception as e_epub: 
        print(f"  File Parser Error: Processing EPUB file '{filepath}': {e_epub}")
    return docs_with_metadata

# --- Main Dispatcher for File Extraction ---

def extract_documents_from_file(filepath: str, relative_path: str) -> list[tuple[str, dict]]:
    """
    Main dispatcher function for extracting text and metadata from a single file.
    It determines the file type and calls the appropriate specialized extraction function.
    """
    # Prepare base metadata applicable to all parts extracted from this file
    base_metadata = {
        "source_file": relative_path, # Relative path from the input documents folder
        "file_type": os.path.splitext(filepath)[1].lower(), # File extension in lowercase
        "file_last_modified": datetime.fromtimestamp(os.path.getmtime(filepath)).isoformat() # ISO format timestamp
    }
    file_type = base_metadata["file_type"]
    docs_with_metadata: List[Tuple[str, Dict[str, Any]]] = [] # Initialize list

    try:
        # Dispatch to the correct extraction helper based on file type
        if file_type in [".txt", ".md", ".py", ".js", ".html", ".css", ".xml", ".sh", ".bat", ".yaml", ".yml", ".ini", ".cfg", ".toml"]:
            docs_with_metadata.extend(_extract_plain_text_like(filepath, base_metadata))
        elif file_type in [".csv", ".parquet", ".xlsx"]:
            docs_with_metadata.extend(_extract_tabular_formats(filepath, base_metadata, file_type))
        elif file_type == ".pdf":
            docs_with_metadata.extend(_extract_pdf_format(filepath, base_metadata))
        elif file_type == ".docx":
            docs_with_metadata.extend(_extract_docx_to_markdown(filepath, base_metadata))
        elif file_type in [".json", ".jsonl"]:
            docs_with_metadata.extend(_extract_json_formats(filepath, base_metadata, file_type))
        elif file_type == ".epub":
            docs_with_metadata.extend(_extract_epub_format(filepath, base_metadata))
        else:
            # Log if the file type is not recognized or supported
            print(f"  File Parser Info: Skipping unsupported file type '{file_type}' for file: {filepath}")
            
    except Exception as e_main_dispatch: 
        # Catch-all for unexpected errors during the dispatch process itself
        print(f"  File Parser Major Error: During extraction dispatch for '{filepath}' (Type: {file_type}): {e_main_dispatch}")
    
    return docs_with_metadata