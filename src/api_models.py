# src/api_models.py
# This module defines Pydantic models for API request and response validation and serialization.
# It helps ensure data consistency and provides clear API contracts.

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any, Union

# Project-specific configuration for default values
from . import ragdoll_config # Assuming ragdoll_config.py is in the same directory (src)

# --- Pydantic Models for API Payloads and Responses ---

class PipelineRunRequest(BaseModel):
    """
    Defines the request body for triggering a data processing pipeline run.
    Defaults are sourced from ragdoll_config.
    """
    docs_folder: str = Field(default="docs_input", description="Path to the folder containing source documents.")
    vector_data_dir: str = Field(default=ragdoll_config.DEFAULT_DATA_SERVICE_URL, description="Directory to save processed data. Note: This default should point to a file path like 'vector_store_data', not a URL. Correcting this to a sensible path default.") # Corrected default value logic below
    overwrite: bool = Field(default=False, description="If set, overwrite existing data in the vector-data-dir.")
    gpu_device_id: int = Field(default=ragdoll_config.DEFAULT_GPU_DEVICE_ID_PROCESSING, description="GPU ID for processing (-1 for CPU).")
    verbose: bool = Field(default=False, description="Enable verbose logging during pipeline execution.")

    embedding_model_name: str = Field(default=ragdoll_config.DEFAULT_PIPELINE_EMBEDDING_MODEL, description="Name or path of the embedding model.")
    embedding_model_type: Optional[str] = Field(default=None, description="Type of embedding model (e.g., 'model2vec', 'sentence-transformers'). Inferred if None.")
    default_embedding_dim: Optional[int] = Field(default=None, description="Expected embedding dimension. Inferred if None.")
    vicinity_metric: Optional[str] = Field(default=None, description="Vicinity store metric (e.g., 'COSINE'). Inferred if None.")

    chunker_type: str = Field(
        default=ragdoll_config.DEFAULT_CHUNKER_TYPE, 
        examples=list(ragdoll_config.CHUNKER_DEFAULTS.keys()), # Provide examples from config
        description="Type of Chonkie chunker to use for splitting documents."
    )
    num_processing_workers: int = Field(default=ragdoll_config.DEFAULT_CHUNK_PROCESSING_WORKERS, description="Number of worker processes for chunking.")
    
    # --- Chonkie Specific Parameters ---
    # Parameters for different Chonkie chunker types, using defaults from ragdoll_config.
    # These correspond to CLI arguments and are mapped to Chonkie-native keys by the orchestrator.

    # Common for SDPM, Semantic
    chonkie_embedding_model: str = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["embedding_model"])
    chonkie_target_chunk_size: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["chunk_size"])
    chonkie_similarity_threshold: Union[float, int, str] = Field(
        default_factory=lambda: ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["threshold"]
    )
    chonkie_min_sentences: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_sentences"])
    
    # For SDPM, Semantic (shared)
    chonkie_mode: str = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["mode"])
    chonkie_similarity_window: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["similarity_window"])
    chonkie_min_chunk_size: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["min_chunk_size"])
    
    # SDPM specific
    chonkie_skip_window: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sdpm"]["skip_window"])

    # Neural specific
    chonkie_neural_model: str = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["model"])
    chonkie_neural_tokenizer: Optional[str] = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["tokenizer"])
    chonkie_neural_stride: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["stride"])
    chonkie_neural_min_chars: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_neural"]["min_characters_per_chunk"])

    # Basic (Recursive, Sentence, Token)
    chonkie_basic_tokenizer: str = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["tokenizer_or_token_counter"])
    chonkie_basic_chunk_size: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["chunk_size"])
    chonkie_basic_min_chars: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_recursive"]["min_characters_per_chunk"])
    chonkie_sentence_overlap: int = Field(default=ragdoll_config.CHUNKER_DEFAULTS["chonkie_sentence"]["chunk_overlap"])

    # Classification parameters
    enable_classification: bool = Field(default=False)
    classifier_model_name: Optional[str] = Field(default=ragdoll_config.DEFAULT_CLASSIFIER_MODEL)
    candidate_labels: Optional[List[str]] = Field(default_factory=lambda: list(ragdoll_config.DEFAULT_CLASSIFICATION_LABELS))
    classification_batch_size: int = Field(default=ragdoll_config.DEFAULT_CLASSIFICATION_BATCH_SIZE)
    
    # Visualization parameters
    prepare_viz_data: bool = Field(default=False)
    viz_output_file: str = Field(default=ragdoll_config.VISUALIZATION_DATA_FILENAME)
    umap_neighbors: int = Field(default=ragdoll_config.DEFAULT_UMAP_NEIGHBORS)
    umap_min_dist: float = Field(default=ragdoll_config.DEFAULT_UMAP_MIN_DIST)
    umap_metric: str = Field(default=ragdoll_config.DEFAULT_UMAP_METRIC)
    
    # Correcting the default for vector_data_dir if it was mis-assigned from a URL constant
    def __init__(self, **data: Any):
        if 'vector_data_dir' not in data or data['vector_data_dir'] == ragdoll_config.DEFAULT_DATA_SERVICE_URL:
            data['vector_data_dir'] = "vector_store_data" # A more sensible default path
        super().__init__(**data)

    @field_validator("chonkie_similarity_threshold")
    @classmethod
    def validate_threshold(cls, value: Union[float, int, str]) -> Union[float, int, str]:
        if isinstance(value, str) and value.lower() not in ["auto", "smart"]:
            try: return float(value)
            except ValueError:
                try: return int(value)
                except ValueError:
                    raise ValueError("chonkie_similarity_threshold as string must be 'auto', 'smart', or a parsable number.")
        return value

class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=ragdoll_config.DEFAULT_RAG_INITIAL_K, ge=1, le=200, description="Number of top results to retrieve.")
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional similarity score threshold.")

class RerankRequestItem(BaseModel): 
    id: str
    text: str
    score: Optional[float] = Field(default=None, description="Original score from search, if available.")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Original metadata of the chunk.")

class RerankRequest(BaseModel):
    query: str
    retrieved_chunks: List[RerankRequestItem] # List of items to be reranked
    top_n_rerank: int = Field(default=ragdoll_config.DEFAULT_RERANKER_TOP_N, ge=1, le=50, description="Number of chunks to return after reranking.")

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    store_loaded: bool
    store_items_count: Optional[int] = None
    store_metadata: Optional[Dict[str, Any]] = None
    reranker_model_name: Optional[str] = None
    query_embedding_model_name: Optional[str] = None
    query_embedding_device: Optional[str] = None

class SearchResultItem(BaseModel):
    id: str
    text: Optional[str] = None # Text might be omitted in some contexts if only IDs/scores are needed
    score: float
    metadata: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]

class ChunkDetailResponse(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]

class VisualizationDataPoint(BaseModel):
    id: str
    x: float # UMAP x-coordinate
    y: float # UMAP y-coordinate
    source_file: Optional[str] = None
    display_source: str # User-friendly name
    classification: Optional[str] = None # Top classification label
    snippet: str # Text snippet

class VisualizationDataResponse(BaseModel):
    data_points: List[VisualizationDataPoint]
    message: Optional[str] = None # Optional message, e.g., if data is partial or filtered

class MessageResponse(BaseModel): # Generic message response
    message: str
    details: Optional[str] = None