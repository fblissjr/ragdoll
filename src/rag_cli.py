# rag_cli.py
import argparse
import json
import readline # For better input experience
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter

import common_utils
from llm_interface import LLMInterface
from data_service_client import DataServiceClient
import rag_core # For format_context_with_citations, construct_llm_messages, etc.

# --- Tokenizer for Context Counting ---
# Using the robust counter from common_utils which handles fallback
def count_tokens_for_llm_context(text_to_count: str) -> int:
    return common_utils.count_tokens_robustly(text_to_count)

def print_rag_answer_and_sources(
    llm_response: str, 
    original_citations: List[rag_core.Citation], 
    data_service_client: DataServiceClient, # Keep for fetching full text if needed
    show_full_source_text: bool = False
):
    print(f"\n🤖 LLM Response:\n{llm_response}")
    
    cited_tags_in_response = rag_core.extract_cited_tags_from_llm_response(llm_response)
    
    if not cited_tags_in_response:
        print("\n--- No sources explicitly cited by the LLM in the format [Source N] ---")
        # Optionally show all sources provided as context:
        # print("\n--- Context Sources Provided to LLM (not explicitly cited in response) ---")
        # print(rag_core.generate_citation_legend(original_citations))
        return

    print("\n--- Cited Sources ---")
    for tag_id in cited_tags_in_response:
        found_citation = next((cit for cit in original_citations if cit.tag_id == tag_id), None)
        if found_citation:
            print(f"[{found_citation.tag_id}] {found_citation.source_display_name}")
            if show_full_source_text:
                # If the citation object has full text from metadata, use it, else fetch
                full_text = found_citation.metadata.get("text_from_search_result") or found_citation.text_preview
                if not full_text or full_text.endswith("..."): # If it was a snippet or missing
                    print("    Fetching full text...")
                    details = data_service_client.get_chunk_details(found_citation.chunk_id)
                    full_text = details.get("text", "Full text not available.") if details else "Error fetching details."
                print(f"    Text: {full_text[:500]}{'...' if len(full_text) > 500 else ''}")
        else:
            print(f"[Source {tag_id}] - Warning: Tag ID cited by LLM not found in original context sources.")

def handle_rag_query(
    query: str,
    data_client: DataServiceClient,
    llm_client: LLMInterface,
    config: argparse.Namespace
):
    print(f"\n🔍 Searching for '{query}' (Initial K: {config.initial_k_retrieval})...")
    search_payload = data_client.search(query, top_k=config.initial_k_retrieval)

    if not search_payload or not search_payload.get("results"):
        print("  No search results from data service."); return

    retrieved_chunks_from_search: List[Dict[str, Any]] = search_payload["results"]
    if not retrieved_chunks_from_search: print("  No relevant chunks found."); return
        
    print(f"  Retrieved {len(retrieved_chunks_from_search)} initial chunks.")
    for i, item in enumerate(retrieved_chunks_from_search[:3]): # Preview top 3
        print(f"    {i+1}. ID: {item['id']}, Score: {item.get('score',0.0):.4f}, Text: {item.get('text', 'N/A')[:70]}...")

    final_chunks_for_context = retrieved_chunks_from_search
    if config.enable_reranking:
        print(f"\n🔄 Reranking top {len(retrieved_chunks_from_search)} chunks via API (Top N after rerank: {config.reranker_top_n})...")
        # Ensure the items passed to rerank API have 'id' and 'text'
        rerank_input = [
            {"id": c["id"], "text": c.get("text", ""), "score": c.get("score", 0.0), "metadata":c.get("metadata")} 
            for c in retrieved_chunks_from_search if c.get("id") and c.get("text")
        ]
        if not rerank_input:
            print("  No valid chunks with text to send for reranking. Using initial search results.")
        else:
            reranked_payload = data_client.rerank(query, rerank_input, config.reranker_top_n)
            if reranked_payload:
                final_chunks_for_context = reranked_payload # API returns list of SearchResultItem-like dicts
                print(f"  Reranking complete. Using {len(final_chunks_for_context)} chunks for context.")
                for i, item in enumerate(final_chunks_for_context[:3]): # Preview top 3 reranked
                     print(f"    {i+1}. ID: {item['id']}, New Score: {item.get('score',0.0):.4f}, Text: {item.get('text', 'N/A')[:70]}...")
            else:
                print("  Reranking via API failed or returned no results. Using initial search results.")
    
    print(f"\n📝 Formulating context (Max LLM context tokens: {config.max_context_tokens})...")
    # Pass the full chunk data including 'text' and 'metadata' for citation generation
    # Add a 'text_from_search_result' field to metadata for print_rag_answer_and_sources to potentially use
    for chunk_data in final_chunks_for_context:
        if "metadata" in chunk_data and chunk_data.get("text"):
            chunk_data["metadata"]["text_from_search_result"] = chunk_data["text"]

    context_str, citations = rag_core.format_context_with_citations(
        final_chunks_for_context, 
        config.max_context_tokens,
        count_tokens_for_llm_context
    )
    if not context_str: print("  Could not formulate context for LLM."); return

    print(f"\n💬 Sending to LLM (Max Gen: {config.llm_max_tokens}, Temp: {config.llm_temperature})...")
    system_prompt_template = config.system_prompt_template or common_utils.DEFAULT_SYSTEM_PROMPT_RAG
    messages = rag_core.construct_llm_messages(query, context_str, system_prompt_template=system_prompt_template)
    
    llm_response = llm_client.generate_chat_completion(
        messages, max_tokens=config.llm_max_tokens, temperature=config.llm_temperature
    )
    if llm_response:
        print_rag_answer_and_sources(llm_response, citations, data_client, config.show_full_source_text)
    else: print("  LLM did not return a response.")

def handle_explore_query(
    query: str, data_client: DataServiceClient, 
    llm_client: LLMInterface, config: argparse.Namespace
):
    print(f"\n🗺️ Exploring data for: '{query}' (Initial K: {config.initial_k_retrieval})...")
    # For explore, fetch more initial chunks if reranking is not enabled, 
    # otherwise reranker_top_n will be used after reranking.
    k_for_explore = config.explore_max_chunks_to_summarize
    if config.enable_reranking:
        k_for_explore = max(config.explore_max_chunks_to_summarize, config.initial_k_retrieval) 
        # Fetch enough for reranker to pick from
    
    search_payload = data_client.search(query, top_k=k_for_explore)
    if not search_payload or not search_payload.get("results"):
        print("  No search results from data service."); return
    
    retrieved_chunks: List[Dict[str, Any]] = search_payload["results"]
    if not retrieved_chunks: print("  No relevant chunks found for exploration."); return

    final_chunks_for_explore = retrieved_chunks
    if config.enable_reranking and retrieved_chunks:
        print(f"\n🔄 Reranking {len(retrieved_chunks)} chunks for exploration via API...")
        rerank_input = [{"id": c["id"], "text": c.get("text",""), "metadata": c.get("metadata")} for c in retrieved_chunks if c.get("text")]
        if rerank_input:
            reranked_payload = data_client.rerank(query, rerank_input, config.explore_max_chunks_to_summarize) # Rerank and keep only N for summary
            if reranked_payload: final_chunks_for_explore = reranked_payload
            else: print("  Explore reranking failed or returned no results. Using initial search results.")
        else: print("  No valid chunks with text to send for explore reranking.")
    
    # Ensure we only take up to explore_max_chunks_to_summarize for the LLM context
    final_chunks_for_explore = final_chunks_for_explore[:config.explore_max_chunks_to_summarize]

    print(f"  Using {len(final_chunks_for_explore)} chunks for LLM exploration summary:")
    explore_context_parts = []
    cited_sources_for_explore: List[rag_core.Citation] = []
    
    for i, chunk_data in enumerate(final_chunks_for_explore):
        chunk_id = chunk_data.get('id', f'unknown_id_{i}'); chunk_text = chunk_data.get('text', '')
        metadata = chunk_data.get('metadata', {}); display_source = metadata.get('display_source_name', chunk_id)
        print(f"    {i+1}. ID: {chunk_id}, Score: {chunk_data.get('score',0.0):.4f}, Text: {chunk_text[:70]}...")
        if chunk_text:
            explore_context_parts.append(f"[Source {i+1}: {display_source}]\n{chunk_text}")
            cited_sources_for_explore.append(rag_core.Citation(
                i+1, display_source, chunk_id, chunk_text[:150]+"...", metadata))
    
    if not explore_context_parts: print("  No text content for LLM exploration."); return

    if config.analyze_retrieved_chunks:
        print("\n📊 Pre-LLM Analysis of Chunks for Exploration:")
        classifications = [item.get("metadata", {}).get("top_label", "N/A") for item in final_chunks_for_explore if item.get("metadata")]
        if classifications:
            classification_counts = Counter(c for c in classifications if c != 'N/A')
            print("  Top Classification Labels in Set:")
            for label, count in classification_counts.most_common(5): print(f"    - {label}: {count}")
        else: print("  No classification data for analysis.")
    
    full_explore_context = "\n\n---\n\n".join(explore_context_parts)
    print(f"\n💡 Asking LLM to explore/summarize context (LLM Temp: {config.explore_llm_summary_temperature})...")
    
    explore_system_prompt = getattr(config, 'explore_system_prompt_template', None) or \
        common_utils.DEFAULT_SYSTEM_PROMPT_EXPLORE
    
    messages = [{"role": "system", "content": explore_system_prompt.format(context=full_explore_context, query=query)},
                {"role": "user", "content": f"Explore the provided context related to: {query}"}]
    
    llm_exploration_output = llm_client.generate_chat_completion(
        messages, max_tokens=config.explore_llm_summary_max_tokens, temperature=config.explore_llm_summary_temperature
    )
    if llm_exploration_output:
        print(f"\n🧠 LLM Exploration Insights:\n{llm_exploration_output}")
        print("\n--- Sources Used for this Exploration ---")
        print(rag_core.generate_citation_legend(cited_sources_for_explore))
    else: print("  LLM returned no exploration insights.")

def main():
    parser = argparse.ArgumentParser(description="RAGdoll CLI: Interact with your processed documents.")
    parser.add_argument("--mode", type=str, choices=["rag", "explore", "status"], default="rag", help="Interaction mode.")
    parser.add_argument("--query", type=str, help="Query to process directly (for rag/explore).")
    
    client_g = parser.add_argument_group('Data Service Configuration')
    client_g.add_argument("--data-service-url", type=str, default=common_utils.DEFAULT_DATA_SERVICE_URL)

    llm_g = parser.add_argument_group('LLM Service Configuration')
    llm_g.add_argument("--llm-api-url", type=str, default=common_utils.DEFAULT_LLM_API_URL)
    llm_g.add_argument("--llm-max-tokens", type=int, default=common_utils.DEFAULT_LLM_MAX_GENERATION_TOKENS)
    llm_g.add_argument("--llm-temperature", type=float, default=common_utils.DEFAULT_LLM_TEMPERATURE)
    llm_g.add_argument("--llm-model-name", type=str, default="mlx-community/Nous-Hermes-2-Mistral-7B-DPO-4bit-MLX", help="Placeholder model name for OpenAI API spec.")

    rag_p_g = parser.add_argument_group('RAG Mode Parameters')
    rag_p_g.add_argument("--initial-k-retrieval", type=int, default=common_utils.DEFAULT_RAG_INITIAL_K)
    rag_p_g.add_argument("--max-context-tokens", type=int, default=common_utils.DEFAULT_RAG_MAX_CONTEXT_TOKENS)
    rag_p_g.add_argument("--enable-reranking", action="store_true", help="Use API to rerank retrieved chunks.")
    # Reranker model name for info only if API does reranking, batch size not needed by client if API call.
    rag_p_g.add_argument("--reranker-model-name", type=str, default=common_utils.DEFAULT_RERANKER_MODEL, help="Info: Reranker model used by server.")
    rag_p_g.add_argument("--reranker-top-n", type=int, default=common_utils.DEFAULT_RERANKER_TOP_N, help="Number of chunks after API reranking.")
    rag_p_g.add_argument("--system-prompt-template", type=str, default=None, help="Custom system prompt for RAG. Use {context} and {query} placeholders.")
    rag_p_g.add_argument("--show-full-source-text", action="store_true", help="Show longer previews of cited texts in RAG mode.")

    explore_g = parser.add_argument_group('Explore Mode Parameters')
    explore_g.add_argument("--explore-max-chunks-to-summarize", type=int, default=common_utils.DEFAULT_EXPLORE_MAX_CHUNKS_TO_SUMMARIZE)
    explore_g.add_argument("--explore-llm-summary-max-tokens", type=int, default=common_utils.DEFAULT_EXPLORE_LLM_SUMMARY_MAX_TOKENS)
    explore_g.add_argument("--explore-llm-summary-temperature", type=float, default=common_utils.DEFAULT_EXPLORE_LLM_SUMMARY_TEMPERATURE)
    explore_g.add_argument("--analyze-retrieved-chunks", action="store_true", help="In explore mode, show pre-LLM analysis of chunks.")
    explore_g.add_argument("--explore-system-prompt-template", type=str, default=None, help="Custom system prompt for Explore. Use {context} and {query}.")

    args = parser.parse_args()
    print(f"\n--- RAGdoll CLI (Mode: {args.mode.upper()}) ---")

    data_client = DataServiceClient(base_url=args.data_service_url)
    llm_client = LLMInterface(api_url=args.llm_api_url, model_name=args.llm_model_name, 
                              default_max_tokens=args.llm_max_tokens, default_temperature=args.llm_temperature)

    status = data_client.get_status()
    if not status: print("Error: Could not connect to Data Service."); return
    print(f"Data Service: {status.get('status')} - Store Loaded: {status.get('store_loaded')}")
    if not status.get('store_loaded') and args.mode != "status":
        print("Error: Vector store not loaded on data service. Run processing pipeline."); return

    if args.mode == "status": return # Status already printed

    if args.query:
        if args.mode == "rag": handle_rag_query(args.query, data_client, llm_client, args)
        elif args.mode == "explore": handle_explore_query(args.query, data_client, llm_client, args)
    else: # Interactive loop
        while True:
            try:
                user_q = input(f"\nEnter query for '{args.mode}' (or 'quit', 'mode rag', 'mode explore', 'mode status'): ")
                if user_q.lower() == 'quit': break
                if user_q.lower().startswith("mode "):
                    new_mode = user_q.lower().split(" ",1)[1].strip()
                    if new_mode in ["rag", "explore", "status"]: args.mode = new_mode; print(f"Switched to mode: {args.mode.upper()}"); continue
                    else: print(f"Unknown mode: {new_mode}. Available: rag, explore, status"); continue
                if not user_q.strip(): continue
                
                if args.mode == "rag": handle_rag_query(user_q, data_client, llm_client, args)
                elif args.mode == "explore": handle_explore_query(user_q, data_client, llm_client, args)
                elif args.mode == "status": status = data_client.get_status(); print(status if status else "Failed to get status.")
            except KeyboardInterrupt: print("\nExiting..."); break
            except Exception as e: print(f"CLI Error: {e}")

if __name__ == "__main__":
    # This makes sure BGE_TOKENIZER_INSTANCE is loaded before interactive loop starts
    if common_utils.TOKEN_COUNT_FALLBACK_ACTIVE:
        print("RAG CLI (startup): Using basic split() for context token counting due to BGE load issue.")
    else:
        print(f"RAG CLI (startup): BGE Tokenizer ({common_utils.BGE_TOKENIZER_INSTANCE.name_or_path if hasattr(common_utils.BGE_TOKENIZER_INSTANCE, 'name_or_path') else 'instance'}) ready for context token counting.")
    main()