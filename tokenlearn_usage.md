# Advanced Guide: Training Custom Embedding Models with Tokenlearn for RAGdoll

RAGdoll is designed to work seamlessly with various embedding models. While it provides defaults, you can significantly enhance retrieval performance, especially for domain-specific content, by preparing custom `StaticModel` instances using the [Tokenlearn](https://github.com/MinishLab/tokenlearn) library. Tokenlearn (version 0.2.0+) allows you to distill knowledge from powerful "teacher" sentence transformers into efficient `StaticModel`s tailored to your data, often without needing manually labeled datasets for this fine-tuning stage.

This guide outlines how to use Tokenlearn to create such a model and then integrate it into your RAGdoll workflow.

## Why Use Tokenlearn for RAGdoll?

*   **Domain Adaptation:** Fine-tune embedding models to better understand the nuances and vocabulary of your specific document collection (e.g., legal, medical, financial texts).
*   **Improved Performance:** Potentially achieve higher accuracy and relevance in RAGdoll's retrieval step with embeddings that are more attuned to your data.
*   **Model Efficiency:** `StaticModel` instances produced by Tokenlearn are typically smaller and faster than their large teacher models, making them ideal for efficient deployment within RAGdoll.
*   **Unsupervised Fine-tuning:** Tokenlearn's primary fine-tuning stage doesn't require labeled pairs; it learns by mimicking the representations of a larger teacher model on your target corpus.

## Workflow Overview

The process involves two main command-line scripts provided by the `tokenlearn` package:
1.  **`tokenlearn.featurize`**: Processes your chosen corpus with a base "teacher" sentence transformer to generate target vector representations (mean token embeddings for passages).
2.  **`tokenlearn.train`**: Takes these features and trains a new `StaticModel`. This involves:
    *   Distilling an initial `StaticModel` from the same base teacher model.
    *   Optionally creating and adding a custom vocabulary from your corpus.
    *   Applying PCA to the target vectors (from featurization).
    *   Fine-tuning the distilled `StaticModel`'s embeddings and token weights to align its passage representations with the PCA-transformed target vectors.

## Step-by-Step Guide

### Prerequisites

1.  **Install RAGdoll and its Dependencies:** Ensure your RAGdoll project is set up, ideally with its virtual environment activated. If `tokenlearn` is not part of RAGdoll's main dependencies, you might need to install it separately or ensure it's included in your development dependencies.
    ```bash
    # If RAGdoll's Makefile handles dev dependencies that include tokenlearn:
    make install-dev 
    # Otherwise, install tokenlearn manually in the RAGdoll environment:
    # source .venv/bin/activate  (or your venv activation command)
    # uv pip install tokenlearn>=0.2.0
    ```

2.  **Choose a Base "Teacher" Sentence Transformer:** This model's knowledge will be distilled. Select a model appropriate for your data's language and domain.
    *   Examples: `baai/bge-base-en-v1.5`, `sentence-transformers/all-mpnet-base-v2`, `paraphrase-multilingual-mpnet-base-v2`.

3.  **Prepare Your Corpus for Featurization:**
    *   **Hugging Face Dataset:** The easiest way is to use a dataset available on the Hugging Face Hub or one loadable via a dataset loading script.
    *   **Local Files:** If you have local text files (e.g., a directory of `.txt` files), you'll need to create a simple Hugging Face dataset loading script to make them accessible to `tokenlearn.featurize`. (Refer to [Hugging Face documentation on loading local data](https://huggingface.co/docs/datasets/loading#local-and-remote-files)).

### Step 1: Featurize Your Corpus with Tokenlearn

This step generates the target embeddings from your corpus using the base model.

*   **Using RAGdoll's Makefile Helper (Recommended for Convenience):**
    1.  Open RAGdoll's main `Makefile`.
    2.  Customize the `TOKENLEARN_*` variables at the top of the "Tokenlearn Model Preparation" section to match your chosen base model, corpus details, and desired parameters. Key variables to set:
        *   `TOKENLEARN_BASE_MODEL`
        *   `TOKENLEARN_CORPUS_HF_PATH` (path or name of your HF dataset)
        *   `TOKENLEARN_CORPUS_HF_NAME` (subset/config name, if any)
        *   `TOKENLEARN_CORPUS_HF_SPLIT`
        *   `TOKENLEARN_CORPUS_TEXT_KEY`
        *   `TOKENLEARN_MAX_MEANS`
        *   `RAGDOLL_TOKENLEARN_FEATURES_DIR` (output directory for features)
    3.  Run the command:
        ```bash
        make tokenlearn-prepare
        ```
        This will create the features in the directory specified by `RAGDOLL_TOKENLEARN_FEATURES_DIR`.

*   **Running `tokenlearn.featurize` Manually:**
    ```bash
    # Activate your RAGdoll virtual environment if not already active
    # source .venv/bin/activate 

    python -m tokenlearn.featurize \
        --model-name "your_chosen_base_model_name" \
        --output-dir "path/to/your_features_output_directory" \
        --dataset-path "your_hf_dataset_path_or_script" \
        --dataset-name "your_dataset_subset_name" \ # Optional
        --dataset-split "train" \
        --key "text_column_in_your_dataset" \
        --max-means 2000000 \  # Adjust based on your needs
        --batch-size 64       # Adjust based on your GPU memory
        # --no-streaming      # If your dataset is small
    ```
    This process can take a long time. Tokenlearn's featurize script can be resumed if interrupted by running the same command again with the same `--output-dir`.

### Step 2: Train the Custom `StaticModel` with Tokenlearn

This step uses the generated features to train your new, specialized `StaticModel`.

*   **Using RAGdoll's Makefile Helper (Recommended):**
    1.  Ensure the `TOKENLEARN_*` variables and `RAGDOLL_TOKENLEARN_FEATURES_DIR` in the `Makefile` are correctly set (or were set for the `tokenlearn-prepare` step). Customize `RAGDOLL_TOKENLEARN_MODEL_DIR`, `TOKENLEARN_PCA_DIMS`, and `TOKENLEARN_VOCAB_SIZE` as needed.
    2.  Run the command:
        ```bash
        make tokenlearn-custom-train
        ```
        This will use the features from `RAGDOLL_TOKENLEARN_FEATURES_DIR` and save the trained model to `RAGDOLL_TOKENLEARN_MODEL_DIR`.

*   **Running `tokenlearn.train` Manually:**
    ```bash
    # Activate your RAGdoll virtual environment

    python -m tokenlearn.train \
        --model-name "your_chosen_base_model_name" \ # Used for tokenizer and initial distillation by Tokenlearn
        --data-path "path/to/your_features_output_directory" \ # From Step 1
        --save-path "path/to/save_your_new_static_model" \
        --pca-dims 256 \         # Or 512, etc. Explained variance is key.
        --vocab-size 30000 \     # Number of custom vocabulary tokens; 0 to disable.
        --device "cuda:0" \      # Or "cpu", "mps"
        # --batch-size 256 \     # Training batch size for tokenlearn.train
        # --lr 1e-3 \            # Learning rate
        # --patience 5           # Early stopping patience
    ```
    Tokenlearn will typically save two model variants in the `--save-path` directory. The **weighted version** is generally recommended for downstream tasks like RAGdoll. Check the Tokenlearn documentation or output logs for the exact naming.

### Step 3: Configure RAGdoll to Use Your Custom Model

Once your Tokenlearn-ed `StaticModel` is trained and saved (e.g., to `$(CURDIR)/custom_models/my_tokenlearn_model`), you need to tell RAGdoll to use it:

1.  **For RAGdoll's Data Processing (`process_data_cli.py` or `/pipeline/run` API):**
    *   When you run the RAGdoll data processing pipeline, specify your custom model's path using the `--embedding-model-name` argument (for CLI) or the `embedding_model_name` field (for API).
    ```bash
    # Example using RAGdoll's CLI
    python -m src.process_data_cli \
        --embedding-model-name "./custom_models/my_tokenlearn_model" \ 
        --vector-data-dir "./vector_store_for_custom_model" \
        --docs-folder "./my_domain_documents" \
        --overwrite \
        # ... other RAGdoll pipeline arguments ...
    ```

2.  **For RAGdoll's Query-Time Embedding (RAG CLI & API Search):**
    *   It's crucial that the model used for embedding queries at runtime matches the one used to embed your documents.
    *   The RAGdoll API service (`main_api.py`) loads its query embedding model based on the metadata from the loaded vector store. If you process your data with the custom model, the vector store's metadata will point to it, and the API should automatically use it for queries.
    *   If you are using the RAGdoll CLI (`rag_cli.py`) and it directly instantiates an embedding model (though it usually relies on the API service), ensure its configuration points to the same custom model. (Currently, `rag_cli.py` calls the API service, which handles query embedding).

**Important Considerations:**

*   **Model Path:** Ensure the path provided to `--embedding-model-name` is correct and accessible by the RAGdoll process.
*   **Consistency:** The model used for document processing (creating the vector store) and query embedding *must* be the same for meaningful similarity search.
*   **`infer_embedding_model_params` in RAGdoll:** RAGdoll's `embedding_module.py` attempts to infer parameters like dimension and metric. For local `StaticModel` paths (like those from Tokenlearn), it tries to read `config.json`. If your Tokenlearn model's `config.json` accurately reflects its dimension, RAGdoll should pick it up.

By following this workflow, you can leverage Tokenlearn to create highly effective, domain-adapted embedding models for your RAGdoll system, leading to improved retrieval accuracy and relevance.