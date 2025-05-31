# process_data_cli.py
import argparse
import os
import common_utils 
import pipeline_orchestrator 

def main_cli():
    parser = argparse.ArgumentParser(description="RAGdoll Data Processing CLI - Orchestrator")
    
    # General Paths & Behavior
    pg = parser.add_argument_group('General Pipeline Settings')
    pg.add_argument("--docs-folder", type=str, default="docs_input")
    pg.add_argument("--vector-data-dir", type=str, default="vector_store_data")
    pg.add_argument("--overwrite", action="store_true", default=False) # Default False for safety
    pg.add_argument("--gpu-device-id", type=int, default=common_utils.DEFAULT_GPU_DEVICE_ID_PROCESSING)
    pg.add_argument("--verbose", "-v", action="store_true", default=False)

    # Embedding Model
    emb_g = parser.add_argument_group('Pipeline Embedding Model (for chunk vectorization)')
    emb_g.add_argument("--embedding-model-name", type=str, default=common_utils.DEFAULT_PIPELINE_EMBEDDING_MODEL)
    emb_g.add_argument("--embedding-model-type", type=str, choices=["model2vec", "sentence-transformers"], default=None)
    emb_g.add_argument("--default-embedding-dim", type=int, default=None)
    emb_g.add_argument("--vicinity-metric", type=str, choices=["COSINE", "INNER_PRODUCT", "EUCLIDEAN"], default=None)

    # Chunker Selection
    chunk_g = parser.add_argument_group('Chunking Strategy')
    chunk_g.add_argument("--chunker-type", type=str, 
                         choices=["semchunk", "chonkie_sdpm", "chonkie_semantic", "chonkie_neural", "chonkie_recursive", "chonkie_sentence", "chonkie_token"], 
                         default=common_utils.DEFAULT_CHUNKER_TYPE)
    chunk_g.add_argument("--num-processing-workers", type=int, default=common_utils.DEFAULT_CHUNK_PROCESSING_WORKERS)
    
    # Semchunk specific
    semchunk_p = parser.add_argument_group('Semchunk Parameters (if chunker-type=semchunk)')
    semchunk_p.add_argument("--semchunk-max-tokens-chunk", type=int, default=common_utils.CHUNKER_DEFAULTS["semchunk"]["max_tokens_chunk"])
    semchunk_p.add_argument("--semchunk-overlap-percent", type=int, default=common_utils.CHUNKER_DEFAULTS["semchunk"]["overlap_percent"])
    
    # Chonkie common (SDPM, Semantic)
    chonkie_s_p = parser.add_argument_group('Chonkie SDPM/Semantic Parameters')
    chonkie_s_p.add_argument("--chonkie-embedding-model", type=str, default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["embedding_model"])
    chonkie_s_p.add_argument("--chonkie-target-chunk-size", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["target_chunk_size"])
    chonkie_s_p.add_argument("--chonkie-similarity-threshold", type=str, default=str(common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["threshold"]), help='Float (0-1), Int (1-100 percentile), or "auto"/"smart".')
    chonkie_s_p.add_argument("--chonkie-min-sentences", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_sentences"])
    chonkie_s_p.add_argument("--chonkie-mode", type=str, choices=["window", "cumulative"], default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["mode"])
    chonkie_s_p.add_argument("--chonkie-similarity-window", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["similarity_window"])
    chonkie_s_p.add_argument("--chonkie-min-chunk-size", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_chunk_size"])

    # Chonkie SDPM specific
    chonkie_sdpm_p = parser.add_argument_group('Chonkie SDPM Specific (if chunker-type=chonkie_sdpm)')
    chonkie_sdpm_p.add_argument("--chonkie-skip-window", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_sdpm"]["skip_window"])
    
    # Chonkie Neural specific
    chonkie_n_p = parser.add_argument_group('Chonkie Neural Specific (if chunker-type=chonkie_neural)')
    chonkie_n_p.add_argument("--chonkie-neural-model", type=str, default=common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["segmentation_model"])
    chonkie_n_p.add_argument("--chonkie-neural-stride", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["stride"])
    chonkie_n_p.add_argument("--chonkie-neural-min-chars", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_neural"]["min_chars_per_chunk"])

    # Chonkie Recursive/Sentence/Token (Basic) specific - these often share tokenizer and chunk_size concepts
    chonkie_b_p = parser.add_argument_group('Chonkie Basic Chunkers (Recursive, Sentence, Token)')
    chonkie_b_p.add_argument("--chonkie-basic-tokenizer", type=str, default=common_utils.CHUNKER_DEFAULTS["chonkie_recursive"]["tokenizer_or_token_counter"])
    chonkie_b_p.add_argument("--chonkie-basic-chunk-size", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_recursive"]["chunk_size"])
    chonkie_b_p.add_argument("--chonkie-basic-min-chars", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_recursive"]["min_characters_per_chunk"])
    # Add overlap for Sentence/Token if they become primary options
    chonkie_b_p.add_argument("--chonkie-sentence-overlap", type=int, default=common_utils.CHUNKER_DEFAULTS["chonkie_sentence"]["chunk_overlap"])


    # Classification
    class_g = parser.add_argument_group('Zero-Shot Classification')
    class_g.add_argument("--enable-classification", action="store_true")
    class_g.add_argument("--classifier-model-name", type=str, default=common_utils.DEFAULT_CLASSIFIER_MODEL)
    class_g.add_argument("--candidate-labels", type=str, nargs="+", default=None)
    class_g.add_argument("--classification-batch-size", type=int, default=common_utils.DEFAULT_CLASSIFICATION_BATCH_SIZE)

    # Visualization
    viz_g = parser.add_argument_group('UMAP Visualization Data') # Same
    viz_g.add_argument("--prepare-viz-data", action="store_true")
    viz_g.add_argument("--viz-output-file", type=str, default=common_utils.VISUALIZATION_DATA_FILENAME)
    viz_g.add_argument("--umap-neighbors", type=int, default=common_utils.DEFAULT_UMAP_NEIGHBORS)
    viz_g.add_argument("--umap-min-dist", type=float, default=common_utils.DEFAULT_UMAP_MIN_DIST)
    viz_g.add_argument("--umap-metric", type=str, default=common_utils.DEFAULT_UMAP_METRIC)
    
    args = parser.parse_args()

    pipeline_config = vars(args).copy()
    if pipeline_config["candidate_labels"] is None: 
        pipeline_config["candidate_labels"] = list(common_utils.DEFAULT_CLASSIFICATION_LABELS)
    
    # Convert chonkie_similarity_threshold to float/int/"auto" if needed
    try:
        pipeline_config["chonkie_similarity_threshold"] = float(args.chonkie_similarity_threshold)
    except ValueError:
        try:
            pipeline_config["chonkie_similarity_threshold"] = int(args.chonkie_similarity_threshold)
        except ValueError:
            if args.chonkie_similarity_threshold.lower() not in ["auto", "smart"]: # "smart" might be for SemanticChunker
                print(f"Warning: Invalid chonkie_similarity_threshold '{args.chonkie_similarity_threshold}'. Using 'auto'.")
                pipeline_config["chonkie_similarity_threshold"] = "auto"
            else:
                 pipeline_config["chonkie_similarity_threshold"] = args.chonkie_similarity_threshold.lower()


    success = pipeline_orchestrator.run_full_processing_pipeline(pipeline_config)
    if success: print("\nCLI: Pipeline completed successfully via orchestrator.")
    else: print("\nCLI: Pipeline failed or aborted by orchestrator.")

if __name__ == "__main__":
    main_cli()