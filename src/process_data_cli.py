# process_data_cli.py
# Command-Line Interface for orchestrating the RAGdoll data processing pipeline.

import argparse # For parsing command-line arguments
import os       # For os.path related checks if any (though mostly handled by orchestrator)

# Project-specific imports
from ragdoll import ragdoll_config # Default configurations
from ragdoll import pipeline_orchestrator # The main pipeline execution logic

def main_cli():
    """
    Sets up and parses command-line arguments for the data processing pipeline,
    then calls the pipeline orchestrator with the gathered configuration.
    """
    parser = argparse.ArgumentParser(
        description="RAGdoll Data Processing CLI - Orchestrates document ingestion, chunking, embedding, and vector store creation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows default values in help message
    )
    
    # --- General Pipeline Settings ---
    # Arguments related to file paths, processing behavior, and resources.
    pg_general = parser.add_argument_group('General Pipeline Settings')
    pg_general.add_argument("--docs-folder", type=str, default="docs_input",
                            help="Path to the folder containing source documents to process.")
    pg_general.add_argument("--vector-data-dir", type=str, default="vector_store_data",
                            help="Directory where processed data and the vector store will be saved.")
    pg_general.add_argument("--overwrite", action="store_true", default=False,
                            help="If set, overwrite existing data in the vector-data-dir.")
    pg_general.add_argument("--gpu-device-id", type=int, default=ragdoll_config.DEFAULT_GPU_DEVICE_ID_PROCESSING,
                            help="GPU device ID to use for processing (e.g., embeddings, classification). -1 for CPU.")
    pg_general.add_argument("--verbose", "-v", action="store_true", default=False,
                            help="Enable verbose logging output during processing.")

    # --- Embedding Model Configuration ---
    # Arguments for selecting and configuring the embedding model used for document chunks.
    pg_embedding = parser.add_argument_group('Pipeline Embedding Model (for chunk vectorization)')
    pg_embedding.add_argument("--embedding-model-name", type=str, default=ragdoll_config.DEFAULT_PIPELINE_EMBEDDING_MODEL,
                              help="Name or path of the sentence-transformer or model2vec model for embeddings.")
    pg_embedding.add_argument("--embedding-model-type", type=str, choices=["model2vec", "sentence-transformers"], default=None,
                              help="Explicitly specify the type of embedding model (model2vec or sentence-transformers). If None, it will be inferred.")
    pg_embedding.add_argument("--default-embedding-dim", type=int, default=None,
                              help="Expected dimension of embeddings. If None, it will be inferred. Useful for validation or if inference fails.")
    pg_embedding.add_argument("--vicinity-metric", type=str, choices=["COSINE", "INNER_PRODUCT", "EUCLIDEAN"], default=None,
                              help="Distance metric for the Vicinity vector store. If None, it's inferred from the embedding model type.")

    # --- Chunking Strategy Configuration ---
    # Arguments for choosing and configuring the text chunking method.
    pg_chunking = parser.add_argument_group('Chunking Strategy')
    pg_chunking.add_argument("--chunker-type", type=str, 
                             choices=list(ragdoll_config.CHUNKER_DEFAULTS.keys()), # Dynamically get choices from config
                             default=ragdoll_config.DEFAULT_CHUNKER_TYPE,
                             help="Type of chunker to use for splitting documents.")
    pg_chunking.add_argument("--num-processing-workers", type=int, default=ragdoll_config.DEFAULT_CHUNK_PROCESSING_WORKERS,
                             help="Number of worker processes for parallel document chunking.")
    
    # Semchunk parameters are removed as semchunk is deprecated.

    # --- Chonkie SDPM/Semantic Chunker Parameters ---
    # These parameters are relevant if chunker-type is chonkie_sdpm or chonkie_semantic.
    pg_chonkie_semantic_dense = parser.add_argument_group('Chonkie SDPM/Semantic Parameters')
    pg_chonkie_semantic_dense.add_argument("--chonkie-embedding-model", type=str, 
                                           default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["embedding_model"],
                                           help="Embedding model used internally by Chonkie SDPM/Semantic chunkers.")
    pg_chonkie_semantic_dense.add_argument("--chonkie-target-chunk-size", type=int, 
                                           default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["chunk_size"],
                                           help="Target chunk size (in tokens) for Chonkie SDPM/Semantic chunkers.")
    pg_chonkie_semantic_dense.add_argument("--chonkie-similarity-threshold", type=str, 
                                           default=str(ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["threshold"]),
                                           help="Similarity threshold for Chonkie SDPM/Semantic. Can be float (0-1), int (1-100 percentile), or 'auto'/'smart'.")
    pg_chonkie_semantic_dense.add_argument("--chonkie-min-sentences", type=int, 
                                           default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_sentences"],
                                           help="Minimum sentences per chunk for Chonkie SDPM/Semantic/Sentence chunkers.")
    pg_chonkie_semantic_dense.add_argument("--chonkie-mode", type=str, choices=["window", "cumulative"], 
                                           default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["mode"],
                                           help="Mode for similarity calculation in Chonkie SDPM/Semantic ('window' or 'cumulative').")
    pg_chonkie_semantic_dense.add_argument("--chonkie-similarity-window", type=int, 
                                           default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["similarity_window"],
                                           help="Similarity window size for Chonkie SDPM/Semantic.")
    pg_chonkie_semantic_dense.add_argument("--chonkie-min-chunk-size", type=int, 
                                           default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_chunk_size"],
                                           help="Minimum final chunk size (tokens) for Chonkie SDPM/Semantic.")

    # --- Chonkie SDPM Specific Parameters ---
    pg_chonkie_sdpm_specific = parser.add_argument_group('Chonkie SDPM Specific Parameters')
    pg_chonkie_sdpm_specific.add_argument("--chonkie-skip-window", type=int, 
                                          default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["skip_window"],
                                          help="Skip window for peak finding in Chonkie SDPM.")
    
    # --- Chonkie Neural Chunker Specific Parameters ---
    pg_chonkie_neural_specific = parser.add_argument_group('Chonkie Neural Specific Parameters')
    pg_chonkie_neural_specific.add_argument("--chonkie-neural-model", type=str, 
                                            default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["model"],
                                            help="Segmentation model for Chonkie Neural chunker.")
    pg_chonkie_neural_specific.add_argument("--chonkie-neural-tokenizer", type=str, 
                                            default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"].get("tokenizer", 
                                                                      ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["model"]),
                                            help="Tokenizer for the Chonkie Neural model (often same as model).")
    pg_chonkie_neural_specific.add_argument("--chonkie-neural-stride", type=int, 
                                            default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["stride"],
                                            help="Stride for Chonkie Neural chunker inference.")
    pg_chonkie_neural_specific.add_argument("--chonkie-neural-min-chars", type=int, 
                                            default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["min_characters_per_chunk"],
                                            help="Minimum characters per chunk for Chonkie Neural.")

    # --- Chonkie Basic Chunker Parameters (Recursive, Sentence, Token) ---
    pg_chonkie_basic = parser.add_argument_group('Chonkie Basic Chunkers (Recursive, Sentence, Token) Parameters')
    pg_chonkie_basic.add_argument("--chonkie-basic-tokenizer", type=str, 
                                  default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["tokenizer_or_token_counter"],
                                  help="Tokenizer or token counter function name (e.g., 'gpt2', 'ragdoll_utils.BGE_TOKENIZER_INSTANCE') for basic Chonkie chunkers.")
    pg_chonkie_basic.add_argument("--chonkie-basic-chunk-size", type=int, 
                                  default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["chunk_size"],
                                  help="Target chunk size (tokens) for basic Chonkie chunkers.")
    pg_chonkie_basic.add_argument("--chonkie-basic-min-chars", type=int, 
                                  default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["min_characters_per_chunk"],
                                  help="Minimum characters per chunk (for Recursive) or per sentence (for Sentence).")
    pg_chonkie_basic.add_argument("--chonkie-sentence-overlap", type=int, 
                                  default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sentence"]["chunk_overlap"], # Default from sentence
                                  help="Chunk overlap (tokens or fraction if Chonkie supports) for Sentence/Token chunkers.")

    # --- Zero-Shot Classification Configuration ---
    pg_classification = parser.add_argument_group('Zero-Shot Classification')
    pg_classification.add_argument("--enable-classification", action="store_true",
                                   help="Enable zero-shot classification of text chunks.")
    pg_classification.add_argument("--classifier-model-name", type=str, default=ragdoll_config.DEFAULT_CLASSIFIER_MODEL,
                                   help="Hugging Face model name for zero-shot classification.")
    pg_classification.add_argument("--candidate-labels", type=str, nargs="+", default=None, # Default is None, orchestrator will use config default if None
                                   help=f"Space-separated list of candidate labels for classification. Defaults to a predefined list in config ({len(ragdoll_config.DEFAULT_CLASSIFICATION_LABELS)} labels).")
    pg_classification.add_argument("--classification-batch-size", type=int, default=ragdoll_config.DEFAULT_CLASSIFICATION_BATCH_SIZE,
                                   help="Batch size for chunk classification.")

    # --- UMAP Visualization Data Generation ---
    pg_visualization = parser.add_argument_group('UMAP Visualization Data')
    pg_visualization.add_argument("--prepare-viz-data", action="store_true",
                                  help="Generate UMAP 2D projection data for visualization.")
    pg_visualization.add_argument("--viz-output-file", type=str, default=ragdoll_config.VISUALIZATION_DATA_FILENAME,
                                  help="Filename for the UMAP visualization JSON data.")
    pg_visualization.add_argument("--umap-neighbors", type=int, default=ragdoll_config.DEFAULT_UMAP_NEIGHBORS,
                                  help="Number of neighbors for UMAP algorithm.")
    pg_visualization.add_argument("--umap-min-dist", type=float, default=ragdoll_config.DEFAULT_UMAP_MIN_DIST,
                                  help="Minimum distance for UMAP algorithm.")
    pg_visualization.add_argument("--umap-metric", type=str, default=ragdoll_config.DEFAULT_UMAP_METRIC,
                                  help="Distance metric for UMAP algorithm (e.g., 'cosine', 'euclidean').")
    
    args = parser.parse_args()

    # Prepare the configuration dictionary to pass to the orchestrator
    pipeline_config = vars(args).copy() # Convert argparse.Namespace to a dictionary

    # Handle default candidate labels if not provided via CLI
    if pipeline_config["candidate_labels"] is None: 
        pipeline_config["candidate_labels"] = list(ragdoll_config.DEFAULT_CLASSIFICATION_LABELS) # Ensure it's a list for orchestrator
    
    # Parse chonkie_similarity_threshold: it can be 'auto', 'smart', a float, or an int (percentile)
    # The orchestrator's _get_specific_chunker_params will pass this string/value to Chonkie,
    # which should handle parsing "auto"/"smart" or numeric types for its 'threshold' parameter.
    # No explicit conversion is strictly needed here if Chonkie handles it, but validation is good.
    sim_thresh_str = args.chonkie_similarity_threshold
    if sim_thresh_str.lower() not in ["auto", "smart"]:
        try:
            # Attempt to convert to float or int if not 'auto' or 'smart'
            # This allows users to pass numbers directly via CLI.
            # The `pipeline_config` will hold the converted numeric type or the original string.
            pipeline_config["chonkie_similarity_threshold"] = float(sim_thresh_str)
        except ValueError:
            try:
                pipeline_config["chonkie_similarity_threshold"] = int(sim_thresh_str)
            except ValueError:
                # If it's not 'auto', 'smart', float, or int, keep the string. Chonkie might error.
                print(f"CLI Warning: chonkie_similarity_threshold '{sim_thresh_str}' is not 'auto', 'smart', or a valid number. Passing as string.")
                pipeline_config["chonkie_similarity_threshold"] = sim_thresh_str
    else:
        pipeline_config["chonkie_similarity_threshold"] = sim_thresh_str.lower()


    # Run the full data processing pipeline using the orchestrator
    success = pipeline_orchestrator.run_full_processing_pipeline(pipeline_config)
    
    if success: 
        print("\nCLI: Pipeline processing completed successfully via orchestrator.")
    else: 
        print("\nCLI: Pipeline processing failed or was aborted by the orchestrator.")

if __name__ == "__main__":
    # This entry point allows the script to be run directly from the command line.
    main_cli()