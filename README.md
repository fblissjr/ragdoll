# RAGdoll: Experimenting with RAG Again

trying to see if rag works better with a more creative interface, static embedding models, better chunking, data modeling, and other experimental stuff.

the system features a modular Python backend service, exposed via FastAPI - cient interactions are available via a big and lofty Python CLI, with a modern interactive frontend potentially coming.

the LLM component is designed to interface with any external OpenAI-compatible API (`mlx-lm` is what I've tested with)

## Features

* **Versatile Data Ingestion:** Supports a wide array of file formats including plain text, Markdown, DOCX (via Pandoc conversion to Markdown), PDF, JSON, JSONL, CSV, Parquet, Excel, and EPUB.
* **Advanced Semantic Chunking:** Utilizes the `chonkie` library for sophisticated text chunking strategies (Token, Sentence, Recursive, SDPM, Semantic, Neural) to maximize semantic coherence.
* **Efficient Vectorization & Retrieval:**
  * Employs fast and effective embedding models (e.g., Model2Vec Potion, Sentence Transformers).
  * Uses [Vicinity](https://github.com/MinishLab/vicinity) as a lightweight, efficient vector store.
* **Reranking:** Improves retrieval relevance with cross-encoder models.
* **Modular Architecture:** Components are designed for easy extension and upgrades.
* **FastAPI Service:** Exposes RESTful APIs for data processing, search, reranking, and fetching chunk details.
* **Comprehensive CLI:**
  * `ragdoll-process`: For running the full data processing pipeline.
  * `ragdoll-cli`: For interactive RAG Q&A and data exploration.
* **Optional Zero-Shot Classification:** Classify chunks into predefined categories during processing.
* **UMAP Visualization:** Generate 2D projections of chunk embeddings for visual exploration.
* **Tokenlearn Integration (Workflow):** Guidance and helper targets for using [Tokenlearn](https://github.com/MinishLab/tokenlearn) to train/fine-tune custom `StaticModel` (Model2Vec) instances for improved embedding quality and domain adaptation.

## Project Structure

```

ragdoll/
├── data/                     # Example directory for source documents for processing
├── custom_models/            # Example directory for custom trained models (e.g., from Tokenlearn)
├── vector_store_output/      # Example directory for pipeline output
├── docs/
│   └── tokenlearn_usage.md   # Guide for using Tokenlearn with RAGdoll
├── LICENSE
├── Makefile                  # Main Makefile for development and common tasks
├── pyproject.toml            # Project metadata and dependencies (for Hatch build system)
├── requirements.txt          # pip freeze output / alternative dependency list
├── src/                      # Main source code for RAGdoll
│   ├── **init**.py
│   ├── api_models.py         # Pydantic models for API
│   ├── api_routers/          # FastAPI routers for different endpoint groups
│   ├── core_logic/           # Core data processing sub-modules (parsing, classification)
│   ├── utils/                # General utility sub-modules (visualization)
│   ├── data_processing_core.py # Orchestrates document prep and chunking
│   ├── data_service_client.py  # Client for RAGdoll API service
│   ├── embedding_module.py   # Handles embedding model loading and generation
│   ├── llm_interface.py      # Client for external LLM services
│   ├── main_api.py           # Main FastAPI application
│   ├── pipeline_orchestrator.py # Orchestrates the full data processing pipeline
│   ├── process_data_cli.py   # CLI for `ragdoll-process` script
│   ├── rag_cli.py            # CLI for `ragdoll-cli` script
│   ├── rag_core.py           # Client-side RAG logic (context, citations)
│   ├── ragdoll_config.py     # Default configurations and constants
│   ├── ragdoll_utils.py      # Common utility functions
│   ├── reranker_module.py    # Reranking logic
│   └── vector_store_manager.py # Manages Vicinity vector store
└── tests/                    # Automated tests
    ├── **init**.py
    ├── conftest.py
    ├── sample_files/         # Sample DOCX, PDF, etc. for testing
    └── test_pipeline_processing.py

```

## Installation & Setup

(Assuming you are using `uv` and the provided `Makefile`)

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/ragdoll.git # Replace with your repo
    cd ragdoll
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    make install-dev 
    ```

    This will create a `.venv` directory, install all necessary packages including development tools (like `pytest`, `ruff`, and `tokenlearn` if added to `dev` dependencies), and set up pre-commit hooks.

3. **(Optional but Recommended for DOCX processing) Install Pandoc:**
    RAGdoll uses Pandoc to convert `.docx` files to Markdown for better structure-aware chunking. Install Pandoc from [pandoc.org](https://pandoc.org/installing.html).

## Running the Data Processing Pipeline

Use the `ragdoll-process` CLI (or `make process-data-example` after configuring paths).

```bash
# Activate virtual environment if not already: source .venv/bin/activate
python -m src.process_data_cli --docs-folder path/to/your/documents --vector-data-dir path/to/your/output --overwrite --verbose
```

Run `python -m src.process_data_cli --help` for all available options.

Key options include:

* `--embedding-model-name`: Specify a Hugging Face model name or a path to a local `StaticModel` (e.g., one trained with Tokenlearn).
* `--chunker-type`: Choose from various `chonkie` chunkers.
* `--enable-classification`: To classify chunks.
* `--prepare-viz-data`: To generate UMAP visualization data.

## Running the API Service

The FastAPI service provides endpoints for search, reranking, etc.

```bash
make run-api
# Or manually:
# uvicorn src.main_api:app --reload --port 8001
```

The API will be available at `http://localhost:8001` (default). See `/docs` for Swagger UI.

## Using the RAGdoll CLI

For interactive RAG and data exploration:

```bash
# Activate virtual environment
python -m src.rag_cli --mode rag 
# Or for explore mode:
# python -m src.rag_cli --mode explore
```

The CLI will connect to the API service (ensure it's running). Run `python -m src.rag_cli --help` for options.

## Training Custom Embedding Models with Tokenlearn

For optimal performance on your specific data, you can train or fine-tune `StaticModel` instances using Tokenlearn. See `docs/tokenlearn_usage.md` for a detailed guide and Makefile targets (`make tokenlearn-prepare`, `make tokenlearn-custom-train`).

# 3rd party packages

relies heavily on some fantastic projects (please refer to their licenses for usage):

* [The Minish Lab's](https://github.com/MinishLab) [vicinity](https://github.com/MinishLab/vicinity): super lightweight low-dependency vector store for nearest neighbor search, with support for different backends and evaluation
* [The Minish Lab's](https://github.com/MinishLab) [model2vec](https://github.com/MinishLab/model2vec): turn any sentence transformer model into a really small static model - this quick POC uses an 8M parameter model, `minishlab/potion-base-8M`, distilled by the `model2vec` team, from `baai/bge-base-en-v1.5`. crazy fast (as is the 32M). for best performance, you really should train it within your domain. check out their work though - they have some innovative approaches that i haven't come across much before.
* text chunking via [semchunk](https://github.com/isaacus-dev/semchunk), and an embedding based one, [chonkie](https://github.com/chonkie-inc/chonkie)
