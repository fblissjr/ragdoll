# src/rag_cli.py
# Command-Line Interface for interacting with the RAGdoll system.

import argparse
import readline # For better input experience
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

# Project-specific imports
from . import ragdoll_config
from . import ragdoll_utils
from . import llm_interface
from . import data_service_client
from . import rag_core # For Citation, context formatting, etc.
from .api_models import RerankRequestItem # For type safety when preparing rerank data

# --- Token Counting ---
def _count_tokens_for_llm_context(text_to_count: str) -> int:
    """Wrapper for token counting, delegates to ragdoll_utils."""
    return ragdoll_utils.count_tokens_robustly(text_to_count)

# --- Output Formatting Helpers ---
def _print_rag_answer_and_sources(
    llm_response_text: str, 
    original_citation_objects: List[rag_core.Citation], 
    data_client_instance: data_service_client.DataServiceClient, 
    show_full_source_text_flag: bool = False
):
    """Prints LLM response and details of cited sources for RAG mode."""
    print(f"\n🤖 LLM Response:\n{llm_response_text}")
    cited_tag_ids = rag_core.extract_cited_tags_from_llm_response(llm_response_text)
    if not cited_tag_ids:
        print("\n--- No sources explicitly cited by the LLM in the format [Source N] ---")
        return

    print("\n--- Cited Sources Details ---")
    for tag_id_num in sorted(list(cited_tag_ids)):
        citation_obj = next((cit for cit in original_citation_objects if cit.tag_id == tag_id_num), None)
        if citation_obj:
            print(f"[{citation_obj.tag_id}] {citation_obj.source_display_name}")
            if show_full_source_text_flag:
                full_text = citation_obj.metadata.get("text_from_search_result", citation_obj.text_preview)
                if full_text.endswith("..."):
                    print("    Fetching full text from service...")
                    chunk_details = data_client_instance.get_chunk_details(citation_obj.chunk_id)
                    full_text = chunk_details.get("text", "Full text not available.") if chunk_details else "Error fetching details."
                print(f"    Text: {full_text[:500]}{'...' if len(full_text) > 500 else ''}")
        else:
            print(f"[Source {tag_id_num}] - Warning: Tag ID {tag_id_num} cited by LLM not found in original context sources.")

def _print_explore_summary_and_sources(
    llm_summary_text: str,
    citation_objects_for_explore: List[rag_core.Citation]
):
    """Prints LLM exploration summary and the legend of sources used."""
    print(f"\n🧠 LLM Exploration Insights:\n{llm_summary_text}")
    print(rag_core.generate_citation_legend(citation_objects_for_explore))


# --- Data Retrieval and Processing Helpers ---
def _fetch_initial_chunks(
    data_client_instance: data_service_client.DataServiceClient, 
    query_text: str, 
    top_k: int
) -> Optional[List[Dict[str, Any]]]:
    """Fetches initial chunks from the data service."""
    print(f"\n🔍 Searching for '{query_text}' (Initial K: {top_k})...")
    search_response = data_client_instance.search(query_text, top_k=top_k)
    if not search_response or not search_response.get("results"):
        print("  No search results from data service for this query."); return None
    
    retrieved_chunks = search_response["results"]
    if not retrieved_chunks: 
        print("  No relevant chunks found by the data service."); return None
        
    print(f"  Retrieved {len(retrieved_chunks)} initial chunks.")
    for i, item in enumerate(retrieved_chunks[:3]): 
        print(f"    {i+1}. ID: {item.get('id', 'N/A')}, Score: {item.get('score',0.0):.4f}, Text: {item.get('text', 'N/A')[:70]}...")
    return retrieved_chunks

def _rerank_chunks_via_api(
    data_client_instance: data_service_client.DataServiceClient, 
    query_text: str, 
    chunks_to_rerank: List[Dict[str, Any]], 
    top_n_after_rerank: int
) -> Optional[List[Dict[str, Any]]]:
    """Reranks chunks using the data service API."""
    print(f"\n🔄 Reranking {len(chunks_to_rerank)} chunks (API will return top {top_n_after_rerank})...")
    
    # Prepare items matching the RerankRequestItem structure expected by the client/server
    rerank_input_payload_items: List[Dict[str, Any]] = []
    for c in chunks_to_rerank:
        if c.get("id") and c.get("text") is not None:
            # Constructing as dicts, data_service_client.rerank will handle the Pydantic model if needed.
            rerank_input_payload_items.append({
                "id": c["id"], 
                "text": c.get("text", ""), 
                "score": c.get("score", 0.0), 
                "metadata": c.get("metadata")
            })
        else:
            print(f"  Skipping chunk for rerank due to missing ID or text: {str(c)[:100]}...")

    if not rerank_input_payload_items:
        print("  No valid chunks with text to send for reranking."); return None
    
    reranked_api_response = data_client_instance.rerank(query_text, rerank_input_payload_items, top_n_after_rerank)
    if reranked_api_response: # This is already List[Dict[str, Any]] (SearchResultItem-like)
        print(f"  Reranking complete. Using {len(reranked_api_response)} chunks.")
        for i, item in enumerate(reranked_api_response[:3]):
             print(f"    {i+1}. ID: {item['id']}, New Score: {item.get('score',0.0):.4f}, Text: {item.get('text', 'N/A')[:70]}...")
        return reranked_api_response
    else:
        print("  Reranking via API failed or returned no results."); return None

def _query_llm_with_context(
    llm_client_instance: llm_interface.LLMInterface,
    query_text: str,
    context_str: str,
    system_prompt_template: str,
    max_gen_tokens: int,
    temperature: float
) -> Optional[str]:
    """Queries the LLM with the given context and prompt."""
    print(f"\n💬 Sending to LLM (Max Gen: {max_gen_tokens}, Temp: {temperature})...")
    messages = rag_core.construct_llm_messages(query_text, context_str, system_prompt_template)
    llm_response = llm_client_instance.generate_chat_completion(
        messages, max_tokens=max_gen_tokens, temperature=temperature
    )
    if not llm_response:
        print("  LLM did not return a response.")
    return llm_response

# --- Main Mode Handlers ---
def handle_rag_query(
    query_text: str,
    data_client: data_service_client.DataServiceClient,
    llm_client: llm_interface.LLMInterface,
    cli_args: argparse.Namespace
):
    """Handles a RAG query by orchestrating search, reranking, context formatting, and LLM interaction."""
    initial_chunks = _fetch_initial_chunks(data_client, query_text, cli_args.initial_k_retrieval)
    if not initial_chunks: return

    final_chunks = initial_chunks
    if cli_args.enable_reranking:
        reranked_chunks = _rerank_chunks_via_api(data_client, query_text, initial_chunks, cli_args.reranker_top_n)
        final_chunks = reranked_chunks if reranked_chunks else initial_chunks # Fallback to initial if rerank fails
    
    print(f"\n📝 Formulating LLM context (Max context tokens: {cli_args.max_context_tokens})...")
    for chunk_data in final_chunks: # Add original text to metadata for Citation object
        if "metadata" in chunk_data and chunk_data.get("text"):
            if chunk_data["metadata"] is None: chunk_data["metadata"] = {}
            chunk_data["metadata"]["text_from_search_result"] = chunk_data["text"]

    context_str, citations = rag_core.format_context_with_citations(
        final_chunks, cli_args.max_context_tokens, _count_tokens_for_llm_context
    )
    if not context_str: 
        print("  Could not formulate context for LLM."); return

    system_prompt = cli_args.system_prompt_template or ragdoll_config.DEFAULT_SYSTEM_PROMPT_RAG
    llm_response = _query_llm_with_context(
        llm_client, query_text, context_str, system_prompt, 
        cli_args.llm_max_tokens, cli_args.llm_temperature
    )
    if llm_response:
        _print_rag_answer_and_sources(llm_response, citations, data_client, cli_args.show_full_source_text)

def handle_explore_query(
    query_text: str, 
    data_client: data_service_client.DataServiceClient, 
    llm_client: llm_interface.LLMInterface, 
    cli_args: argparse.Namespace
):
    """Handles an Explore mode query."""
    num_to_retrieve = cli_args.initial_k_retrieval
    if cli_args.enable_reranking:
        num_to_retrieve = max(cli_args.initial_k_retrieval, cli_args.explore_max_chunks_to_summarize * 2) # Fetch more for reranker
    
    initial_chunks = _fetch_initial_chunks(data_client, query_text, num_to_retrieve)
    if not initial_chunks: return

    chunks_for_summary = initial_chunks
    if cli_args.enable_reranking:
        reranked = _rerank_chunks_via_api(data_client, query_text, initial_chunks, cli_args.explore_max_chunks_to_summarize)
        chunks_for_summary = reranked if reranked else initial_chunks[:cli_args.explore_max_chunks_to_summarize]
    else:
        chunks_for_summary = initial_chunks[:cli_args.explore_max_chunks_to_summarize]

    if not chunks_for_summary:
        print("  No chunks selected for LLM exploration summary."); return

    print(f"  Using {len(chunks_for_summary)} chunks for LLM exploration:")
    explore_context_parts = []
    citations_for_explore: List[rag_core.Citation] = []
    
    for i, chunk_data in enumerate(chunks_for_summary):
        chunk_id = chunk_data.get('id', f'exp_unknown_{i}'); chunk_text = chunk_data.get('text', '')
        metadata = chunk_data.get('metadata', {}); display_source = metadata.get('display_source_name', ragdoll_utils.generate_display_source_name(metadata, i))
        print(f"    {i+1}. ID: {chunk_id}, Score: {chunk_data.get('score',0.0):.4f}, Text: {chunk_text[:70]}...")
        if chunk_text:
            explore_context_parts.append(f"[Source {i+1}: {display_source}]\n{chunk_text}")
            citations_for_explore.append(rag_core.Citation(i+1, display_source, chunk_id, chunk_text[:150]+"...", metadata))
    
    if not explore_context_parts: print("  No text content for LLM exploration."); return

    if cli_args.analyze_retrieved_chunks:
        print("\n📊 Pre-LLM Analysis of Chunks for Exploration:")
        classifications = [item.get("metadata", {}).get("top_label", "N/A") for item in chunks_for_summary if item.get("metadata")]
        if classifications:
            counts = Counter(c for c in classifications if c != 'N/A')
            print("  Top Classification Labels:"); [print(f"    - {lbl}: {cnt}") for lbl, cnt in counts.most_common(5)]
        else: print("  No classification data for analysis.")
    
    full_explore_context = "\n\n---\n\n".join(explore_context_parts)
    explore_prompt = cli_args.explore_system_prompt_template or ragdoll_config.DEFAULT_SYSTEM_PROMPT_EXPLORE
    
    # Construct messages for explore mode (different system prompt)
    explore_messages = [
        {"role": "system", "content": explore_prompt.format(context=full_explore_context, query=query_text)},
        {"role": "user", "content": f"Please provide exploration insights for the context related to: {query_text}"}
    ]
    
    llm_exploration_output = llm_client.generate_chat_completion(
        explore_messages, 
        max_tokens=cli_args.explore_llm_summary_max_tokens, 
        temperature=cli_args.explore_llm_summary_temperature
    )
    if llm_exploration_output:
        _print_explore_summary_and_sources(llm_exploration_output, citations_for_explore)
    else: 
        print("  LLM returned no exploration insights.")

# --- Main CLI Function ---
def main():
    """Main function for the RAGdoll CLI."""
    # (Argparse setup remains the same as your previous version, ensuring defaults come from ragdoll_config)
    parser = argparse.ArgumentParser(description="RAGdoll CLI", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, choices=["rag", "explore", "status"], default="rag", help="Interaction mode.")
    parser.add_argument("--query", type=str, help="Query to process directly.")
    
    client_g = parser.add_argument_group('Service Connection Configuration')
    client_g.add_argument("--data-service-url", type=str, default=ragdoll_config.DEFAULT_DATA_SERVICE_URL)
    client_g.add_argument("--llm-api-url", type=str, default=ragdoll_config.DEFAULT_LLM_API_URL)

    llm_g = parser.add_argument_group('LLM Interaction Parameters')
    llm_g.add_argument("--llm-max-tokens", type=int, default=ragdoll_config.DEFAULT_LLM_MAX_GENERATION_TOKENS)
    llm_g.add_argument("--llm-temperature", type=float, default=ragdoll_config.DEFAULT_LLM_TEMPERATURE)
    llm_g.add_argument("--llm-model-name", type=str, default="ragdoll/cli-llm-default")

    rag_p_g = parser.add_argument_group('RAG Mode Parameters')
    rag_p_g.add_argument("--initial-k-retrieval", type=int, default=ragdoll_config.DEFAULT_RAG_INITIAL_K)
    rag_p_g.add_argument("--max-context-tokens", type=int, default=ragdoll_config.DEFAULT_RAG_MAX_CONTEXT_TOKENS)
    rag_p_g.add_argument("--enable-reranking", action="store_true", default=False)
    rag_p_g.add_argument("--reranker-top-n", type=int, default=ragdoll_config.DEFAULT_RERANKER_TOP_N)
    rag_p_g.add_argument("--system-prompt-template", type=str, default=None)
    rag_p_g.add_argument("--show-full-source-text", action="store_true", default=False)

    explore_g = parser.add_argument_group('Explore Mode Parameters')
    explore_g.add_argument("--explore-max-chunks-to-summarize", type=int, default=ragdoll_config.DEFAULT_EXPLORE_MAX_CHUNKS_TO_SUMMARIZE)
    explore_g.add_argument("--explore-llm-summary-max-tokens", type=int, default=ragdoll_config.DEFAULT_EXPLORE_LLM_SUMMARY_MAX_TOKENS)
    explore_g.add_argument("--explore-llm-summary-temperature", type=float, default=ragdoll_config.DEFAULT_EXPLORE_LLM_SUMMARY_TEMPERATURE)
    explore_g.add_argument("--analyze-retrieved-chunks", action="store_true", default=False)
    explore_g.add_argument("--explore-system-prompt-template", type=str, default=None)

    args = parser.parse_args()
    print(f"\n--- RAGdoll CLI (Mode: {args.mode.upper()}) ---")

    data_client = data_service_client.DataServiceClient(base_url=args.data_service_url)
    llm_client = llm_interface.LLMInterface(
        api_url=args.llm_api_url, model_name=args.llm_model_name, 
        default_max_tokens=args.llm_max_tokens, default_temperature=args.llm_temperature
    )

    status = data_client.get_status()
    if not status: print("Error: Could not connect to Data Service."); return
    
    print(f"Data Service Status: {status.get('status', 'Unknown')} - Store Loaded: {status.get('store_loaded', False)}")
    if not status.get('store_loaded', False) and args.mode != "status":
        print("Error: Vector store not loaded on data service."); return

    if args.mode == "status": 
        if status: # Print more details if status was fetched
            print(f"  Reranker (Server): {status.get('reranker_model_name', 'N/A')}")
            print(f"  Query Embedding Model (Server): {status.get('query_embedding_model_name', 'N/A')}")
            print(f"  Query Embedding Device (Server): {status.get('query_embedding_device', 'N/A')}")
            store_meta = status.get('store_metadata')
            if store_meta:
                print("  Store Metadata (Preview):")
                for k, v_item in store_meta.items():
                    if isinstance(v_item, (dict, list)) and len(str(v_item)) > 100: print(f"    {k}: (Preview of complex type)")
                    else: print(f"    {k}: {v_item}")
        return

    if args.query:
        if args.mode == "rag": handle_rag_query(args.query, data_client, llm_client, args)
        elif args.mode == "explore": handle_explore_query(args.query, data_client, llm_client, args)
    else:
        print(f"Entering interactive '{args.mode}' mode. Type 'quit' to exit, or 'mode [new_mode]' to switch.")
        while True:
            try:
                user_q = input(f"\n[{args.mode.upper()}] Query: ")
                if user_q.lower() == 'quit': break
                if user_q.lower().startswith("mode "):
                    new_mode = user_q.lower().split(" ",1)[1].strip()
                    if new_mode in ["rag", "explore", "status"]: 
                        args.mode = new_mode; print(f"Switched to mode: {args.mode.upper()}"); 
                        if args.mode == "status": 
                             status_loop = data_client.get_status()
                             print(status_loop if status_loop else "Failed to get status.")
                        continue
                    else: print(f"Unknown mode: '{new_mode}'. Options: rag, explore, status"); continue
                if not user_q.strip(): continue
                
                if args.mode == "rag": handle_rag_query(user_q, data_client, llm_client, args)
                elif args.mode == "explore": handle_explore_query(user_q, data_client, llm_client, args)
            except KeyboardInterrupt: print("\nExiting..."); break
            except Exception as e: print(f"CLI Error: {e}")

if __name__ == "__main__":
    if ragdoll_utils.TOKEN_COUNT_FALLBACK_ACTIVE or ragdoll_utils.BGE_TOKENIZER_INSTANCE is None:
        print("RAG CLI (Startup): Using basic split() for token counting (BGE Tokenizer issue).")
    else:
        tokenizer_name = getattr(ragdoll_utils.BGE_TOKENIZER_INSTANCE, 'name_or_path', 'BGE_INSTANCE')
        print(f"RAG CLI (Startup): BGE Tokenizer ('{tokenizer_name}') ready.")
    main()