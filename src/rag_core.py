# rag_core.py
import os
import json
import re
from typing import List, Dict, Optional, Tuple, Any

import common_utils 
# For this client-side module, embedding_module and reranker_module are typically not
# directly used if all processing is on the FastAPI service.
# However, if query embedding or reranking were to happen client-side, they'd be needed.
# For now, interaction is via data_service_client.

class Citation:
    def __init__(self, tag_id: int, source_display_name: str, chunk_id: str, text_preview: str, metadata: Dict):
        self.tag_id = tag_id
        self.source_display_name = source_display_name
        self.chunk_id = chunk_id
        self.text_preview = text_preview
        self.metadata = metadata

    def __repr__(self):
        return f"[{self.tag_id}] {self.source_display_name}"

def format_context_with_citations(
    retrieved_chunks: List[Dict[str, Any]], # Chunks usually from SearchResultItem or reranker
    max_context_tokens: int,
    tokenizer_for_counting # This should be a callable that takes text and returns token count
) -> Tuple[str, List[Citation]]:
    
    context_str = ""
    citations: List[Citation] = []
    current_token_count = 0
    citation_idx_counter = 1

    print(f"\nRAG Core: Formatting context with citations. Max tokens: {max_context_tokens}")
    for chunk_data in retrieved_chunks:
        chunk_id = chunk_data.get('id', 'unknown_id')
        chunk_text = chunk_data.get('text', '')
        metadata = chunk_data.get('metadata', {})
        display_source = metadata.get('display_source_name', chunk_id) # Use the pre-generated display name
        
        if not chunk_text: continue

        # Estimate token count for this chunk + citation tag
        # Tag format: "[Source N]\n\n<chunk_text>\n\n"
        # This is a rough estimate; precise tokenization depends on the LLM.
        # Using the BGE tokenizer for a consistent (though not LLM-identical) count.
        citation_tag = f"[Source {citation_idx_counter}]"
        formatted_chunk = f"{citation_tag}\n\n{chunk_text}\n\n"
        
        chunk_token_count = tokenizer_for_counting(formatted_chunk)

        if current_token_count + chunk_token_count <= max_context_tokens:
            context_str += formatted_chunk
            citations.append(Citation(
                tag_id=citation_idx_counter,
                source_display_name=display_source,
                chunk_id=chunk_id,
                text_preview=chunk_text[:150] + "...",
                metadata=metadata
            ))
            current_token_count += chunk_token_count
            citation_idx_counter += 1
            print(f"  Added: {display_source} ({chunk_token_count} tokens). Total context: {current_token_count} tokens.")
        else:
            print(f"  Skipped: {display_source} ({chunk_token_count} tokens) - Exceeds max context. Current: {current_token_count}")
            break
    
    print(f"RAG Core: Final context token count (estimated): {current_token_count} tokens, {len(citations)} sources.")
    return context_str.strip(), citations

def construct_llm_messages(query: str, context: str, system_prompt_template: Optional[str] = None) -> List[Dict[str, str]]:
    if system_prompt_template:
        system_message = system_prompt_template.format(context=context)
    else:
        system_message = f"""You are a helpful AI assistant. Answer the user's question based ONLY on the provided context.
Your answer must be grounded in the information from the context.
If the context does not contain enough information to answer the question, state that and do not attempt to answer.
Cite relevant sources from the context by including their citation tag (e.g., [Source N]) at the end of sentences or paragraphs that use information from that source.
Context:
---
{context}
---
"""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": query}
    ]
    return messages

def generate_citation_legend(citations: List[Citation]) -> str:
    if not citations: return "No sources cited."
    legend_parts = ["\n\n--- Sources ---"]
    for cit in sorted(citations, key=lambda c: c.tag_id): # Ensure sorted by tag_id
        legend_parts.append(f"[{cit.tag_id}] {cit.source_display_name}")
        # Optionally add more details from cit.metadata if desired
        # e.g., if 'top_label' is present and useful:
        # top_label = cit.metadata.get('top_label')
        # if top_label and top_label != 'N/A':
        #     legend_parts[-1] += f" (Topic: {top_label})"
    return "\n".join(legend_parts)

def extract_cited_tags_from_llm_response(llm_response: str) -> List[int]:
    # Finds [Source N] tags
    # Regex to find "[Source" followed by one or more digits and then "]"
    pattern = r"\[Source\s*(\d+)\]" 
    cited_tags = set()
    for match in re.finditer(pattern, llm_response):
        try:
            tag_number = int(match.group(1))
            cited_tags.add(tag_number)
        except ValueError:
            # Should not happen if regex is correct
            print(f"Warning: Could not parse tag number from match: {match.group(0)}")
    return sorted(list(cited_tags))