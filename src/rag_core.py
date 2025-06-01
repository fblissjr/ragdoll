# src/rag_core.py
# Core client-side RAG logic: context formatting, citation handling, LLM prompt construction.

import re 
from typing import List, Dict, Optional, Tuple, Any, Callable 

# Project-specific imports
from . import ragdoll_utils  # For token counting, display name generation
from . import ragdoll_config # For default system prompts

class Citation:
    """Represents a cited source chunk in the RAG context."""
    def __init__(self, tag_id: int, source_display_name: str, chunk_id: str, text_preview: str, metadata: Dict[str, Any]):
        self.tag_id: int = tag_id 
        self.source_display_name: str = source_display_name
        self.chunk_id: str = chunk_id
        self.text_preview: str = text_preview 
        self.metadata: Dict[str, Any] = metadata

    def __repr__(self) -> str:
        return f"[{self.tag_id}] {self.source_display_name} (ID: {self.chunk_id})"

def format_context_with_citations(
    retrieved_chunks: List[Dict[str, Any]], 
    max_context_tokens: int, 
    token_counter_func: Callable[[str], int] 
) -> Tuple[str, List[Citation]]:
    """
    Formats retrieved chunks into a context string with [Source N] tags for the LLM,
    respecting token limits.
    """
    context_parts: List[str] = []
    citations_for_context: List[Citation] = []
    current_total_token_count: int = 0
    citation_tag_counter: int = 1 

    print(f"\nRAG Core: Formatting LLM context. Max tokens: {max_context_tokens}")
    
    for chunk_data in retrieved_chunks:
        chunk_id = chunk_data.get('id', f'unknown_id_ctx_{citation_tag_counter}')
        chunk_text = chunk_data.get('text', '')
        metadata = chunk_data.get('metadata', {})
        
        # Use pre-generated display_source_name or generate one.
        # chunk_order_in_doc_part might not be in metadata if not set during chunking.
        # The citation_tag_counter-1 serves as a proxy for chunk order in this specific context list.
        display_source = metadata.get('display_source_name', 
                                      ragdoll_utils.generate_display_source_name(metadata, citation_tag_counter - 1))
        
        if not chunk_text.strip(): 
            print(f"  Skipped (empty text): Chunk ID '{chunk_id}'.")
            continue

        citation_tag = f"[Source {citation_tag_counter}]"
        formatted_chunk = f"{citation_tag}\n\n{chunk_text}\n\n" # Newlines for LLM readability
        chunk_token_estimate = token_counter_func(formatted_chunk)

        if current_total_token_count + chunk_token_estimate <= max_context_tokens:
            context_parts.append(formatted_chunk)
            citations_for_context.append(Citation(
                tag_id=citation_tag_counter,
                source_display_name=display_source,
                chunk_id=chunk_id,
                text_preview=chunk_text[:150] + ("..." if len(chunk_text) > 150 else ""),
                metadata=metadata
            ))
            current_total_token_count += chunk_token_estimate
            citation_tag_counter += 1
            # print(f"  Added: '{display_source}' (~{chunk_token_estimate} tokens). Total: ~{current_total_token_count}") # Verbose
        else:
            print(f"  Skipped (token limit): '{display_source}' (~{chunk_token_estimate} tokens). Context full.")
            break 
    
    final_context_string = "".join(context_parts).strip()
    print(f"RAG Core: Final context: ~{current_total_token_count} tokens, {len(citations_for_context)} sources.")
    return final_context_string, citations_for_context

def construct_llm_messages(query: str, context_str: str, system_prompt_template: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Constructs messages for the LLM API (system message with context, user query).
    """
    actual_system_prompt = system_prompt_template or ragdoll_config.DEFAULT_SYSTEM_PROMPT_RAG
    
    try:
        # Attempt to format with both context and query, as some templates might use query in system prompt too.
        system_message_content = actual_system_prompt.format(context=context_str, query=query)
    except KeyError: # If {query} is not in the system_prompt_template (which is common)
        try:
            system_message_content = actual_system_prompt.format(context=context_str)
        except KeyError as e_sys_prompt:
            print(f"RAG Core Warning: System prompt template error (missing {{context}}?): {e_sys_prompt}. Using default prompt structure.")
            # Fallback to ensure context is always included if default prompt is used.
            system_message_content = f"{ragdoll_config.DEFAULT_SYSTEM_PROMPT_RAG.split('{context}')[0]}{context_str}{ragdoll_config.DEFAULT_SYSTEM_PROMPT_RAG.split('{context}')[1] if len(ragdoll_config.DEFAULT_SYSTEM_PROMPT_RAG.split('{context}')) > 1 else ''}"


    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": query}
    ]
    return messages

def generate_citation_legend(citations: List[Citation]) -> str:
    """Generates a human-readable list of cited sources."""
    if not citations: return "No sources were cited in the LLM context."
    
    legend_parts = ["\n--- Sources Referenced in LLM Context ---"]
    for cit in sorted(citations, key=lambda c: c.tag_id):
        legend_entry = f"[{cit.tag_id}] {cit.source_display_name}"
        # Optional: Add more details from metadata if desired
        # classification_label = cit.metadata.get('top_label')
        # if classification_label and classification_label != 'N/A':
        #     legend_entry += f" (Topic: {classification_label})"
        legend_parts.append(legend_entry)
    return "\n".join(legend_parts)

def extract_cited_tags_from_llm_response(llm_response: str) -> List[int]:
    """Extracts unique numerical IDs from [Source N] tags in LLM response."""
    citation_pattern = r"\[Source\s*(\d+)\]" 
    cited_tags_set = set()
    for match in re.finditer(citation_pattern, llm_response, re.IGNORECASE):
        try:
            tag_number = int(match.group(1))
            cited_tags_set.add(tag_number)
        except ValueError:
            print(f"RAG Core Warning: Could not parse tag number from: {match.group(0)}")
    return sorted(list(cited_tags_set))