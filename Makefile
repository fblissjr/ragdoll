# ragdoll/Makefile

# --- Variables ---
# UV should be on your system PATH (e.g., installed via curl script or pipx)
UV := uv

# Default virtual environment directory name
VENV_DIR ?= .venv

# --- Environment Setup ---
.PHONY: venv install-dev install help

venv:
	@echo ">>> Creating virtual environment in $(VENV_DIR) using uv..."
	$(UV) venv $(VENV_DIR) # Uses system default python, or you can add --python 3.x
	@echo ">>> Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"
	@echo ">>> To install dependencies, run: make install-dev"

install-dev: venv
	@echo ">>> Installing all dependencies (including dev extras) into $(VENV_DIR) using uv..."
	# uv sync will use the .venv in the current directory by default
	$(UV) sync --all-extras --python $(VENV_DIR)/bin/python # Be explicit about python for sync
	@echo ">>> Installing pre-commit hooks..."
	$(UV) run pre-commit install --python $(VENV_DIR)/bin/python # Tell pre-commit which python to use
	@echo ">>> Dependencies and pre-commit hooks installed."

install: venv
	@echo ">>> Installing production dependencies into $(VENV_DIR) using uv..."
	$(UV) sync --python $(VENV_DIR)/bin/python
	@echo ">>> Production dependencies installed."

# --- Code Quality & Formatting ---
.PHONY: lint format check-types fix-code

lint:
	@echo ">>> Running linters (Ruff check) on src and tests..."
	$(UV) run ruff check src tests

format:
	@echo ">>> Formatting code (Ruff format) for src and tests..."
	$(UV) run ruff format src tests

check-types:
	@echo ">>> Running static type checks (MyPy) on src..."
	$(UV) run mypy src

fix-code:
	@echo ">>> Running pre-commit hooks to fix files..."
	$(UV) run pre-commit run --all-files

# --- Testing ---
.PHONY: test test-cov

test:
	@echo ">>> Running tests (pytest)..."
	$(UV) run pytest tests/

test-cov:
	@echo ">>> Running tests with coverage (pytest-cov)..."
	$(UV) run pytest --cov=src --cov-report=term-missing --cov-report=html tests/
	@echo ">>> Coverage report generated in htmlcov/ directory."

# --- RAGdoll Application Specific Tasks ---
.PHONY: run-api process-data-example run-cli

run-api:
	@echo ">>> Starting RAGdoll API service (Uvicorn)..."
	$(UV) run uvicorn src.main_api:app --reload --host 0.0.0.0 --port 8001

process-data-example:
	@echo ">>> Running RAGdoll data processing pipeline (example configuration)..."
	$(UV) run python -m src.process_data_cli \
		--docs-folder ./sample_docs_for_processing \
		--vector-data-dir ./vector_store_output \
		--overwrite \
		--chunker-type chonkie_sdpm \
		--embedding-model-name "sentence-transformers/all-MiniLM-L6-v2" \
		--verbose
	@echo ">>> Example data processing finished."

run-cli:
	@echo ">>> Starting RAGdoll Interactive CLI..."
	$(UV) run python -m src.rag_cli

# --- Tokenlearn Model Preparation (Helper Targets) ---
TOKENLEARN_BASE_MODEL ?= "baai/bge-base-en-v1.5"
TOKENLEARN_CORPUS_HF_PATH ?= "allenai/c4" 
TOKENLEARN_CORPUS_HF_NAME ?= "en"
TOKENLEARN_CORPUS_HF_SPLIT ?= "train"
TOKENLEARN_CORPUS_TEXT_KEY ?= "text"
TOKENLEARN_MAX_MEANS ?= 1000000 
TOKENLEARN_FEATURIZE_BATCH_SIZE ?= 32
TOKENLEARN_PCA_DIMS ?= 256
TOKENLEARN_VOCAB_SIZE ?= 30000 
TOKENLEARN_TRAIN_DEVICE ?= "cpu" 

RAGDOLL_TOKENLEARN_FEATURES_DIR := $(CURDIR)/data_tokenlearn/features_$(subst /,-,$(TOKENLEARN_BASE_MODEL))
RAGDOLL_TOKENLEARN_MODEL_DIR := $(CURDIR)/custom_models/tokenlearned_$(subst /,-,$(TOKENLEARN_BASE_MODEL))_pca$(TOKENLEARN_PCA_DIMS)

.PHONY: tokenlearn-prepare tokenlearn-custom-train tokenlearn-custom-clean

tokenlearn-prepare:
	@echo ">>> Preparing features for custom Tokenlearn model..."
	@mkdir -p $(RAGDOLL_TOKENLEARN_FEATURES_DIR)
	$(UV) run python -m tokenlearn.featurize \
		--model-name "$(TOKENLEARN_BASE_MODEL)" \
		--output-dir "$(RAGDOLL_TOKENLEARN_FEATURES_DIR)" \
		--dataset-path "$(TOKENLEARN_CORPUS_HF_PATH)" \
		--dataset-name "$(TOKENLEARN_CORPUS_HF_NAME)" \
		--dataset-split "$(TOKENLEARN_CORPUS_HF_SPLIT)" \
		--key "$(TOKENLEARN_CORPUS_TEXT_KEY)" \
		--max-means $(TOKENLEARN_MAX_MEANS) \
		--batch-size $(TOKENLEARN_FEATURIZE_BATCH_SIZE)
	@echo ">>> Tokenlearn featurization complete."

tokenlearn-custom-train: $(RAGDOLL_TOKENLEARN_FEATURES_DIR) 
	@echo ">>> Training custom embedding model with Tokenlearn..."
	@mkdir -p $(RAGDOLL_TOKENLEARN_MODEL_DIR)
	$(UV) run python -m tokenlearn.train \
		--model-name "$(TOKENLEARN_BASE_MODEL)" \
		--data-path "$(RAGDOLL_TOKENLEARN_FEATURES_DIR)" \
		--save-path "$(RAGDOLL_TOKENLEARN_MODEL_DIR)" \
		--pca-dims $(TOKENLEARN_PCA_DIMS) \
		--vocab-size $(TOKENLEARN_VOCAB_SIZE) \
		--device "$(TOKENLEARN_TRAIN_DEVICE)"
	@echo ">>> Tokenlearn training complete."

$(RAGDOLL_TOKENLEARN_FEATURES_DIR):
	@echo "Tokenlearn features directory '$(RAGDOLL_TOKENLEARN_FEATURES_DIR)' not found."
	@echo "Please run 'make tokenlearn-prepare' first."
	@exit 1

tokenlearn-custom-clean:
	@echo ">>> Cleaning Tokenlearn generated data..."
	@rm -rf "$(RAGDOLL_TOKENLEARN_FEATURES_DIR)"
	@rm -rf "$(RAGDOLL_TOKENLEARN_MODEL_DIR)"
	@echo ">>> Tokenlearn custom data cleaned."

# --- General Cleanup Targets ---
.PHONY: clean clean-pyc clean-build

clean-pyc:
	@echo ">>> Removing Python cache files (__pycache__, .pyc, .pyo)..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

clean-build:
	@echo ">>> Removing build artifacts (build/, dist/, .eggs/, *.egg-info, *.egg)..."
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.egg" -delete

clean: clean-pyc clean-build tokenlearn-custom-clean
	@echo ">>> Removing test cache (.pytest_cache) and coverage data (htmlcov/, .coverage)..."
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -f .coverage
	@echo ">>> Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo ">>> Full project cleanup complete."

default: help

help:
	@echo "RAGdoll Project Makefile"
	@echo "--------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Environment & Installation:"
	@echo "  venv          - Create Python virtual environment using uv."
	@echo "  install-dev   - Install all dependencies including development tools (requires venv)."
	@echo "  install       - Install production dependencies (requires venv)."
	@echo ""
	@echo "Code Quality:"
	@echo "  lint          - Run linters (Ruff)."
	@echo "  format        - Format code (Ruff format)."
	@echo "  check-types   - Run static type checks (MyPy)."
	@echo "  fix-code      - Run pre-commit hooks to auto-fix files."
	@echo ""
	@echo "Testing:"
	@echo "  test          - Run automated tests with pytest."
	@echo "  test-cov      - Run tests with code coverage report."
	@echo ""
	@echo "RAGdoll Application:"
	@echo "  run-api       - Start the RAGdoll FastAPI service."
	@echo "  process-data-example - Run data processing pipeline with example settings."
	@echo "  run-cli       - Start the RAGdoll interactive CLI."
	@echo ""
	@echo "Tokenlearn Model Preparation (for custom embedding models):"
	@echo "  tokenlearn-prepare        - Run Tokenlearn featurization (customize variables in Makefile)."
	@echo "  tokenlearn-custom-train   - Train a custom model using Tokenlearn (depends on preparation)."
	@echo "  tokenlearn-custom-clean   - Clean up Tokenlearn generated features and model."
	@echo ""
	@echo "General Cleanup:"
	@echo "  clean-pyc     - Remove Python cache files."
	@echo "  clean-build   - Remove build artifacts."
	@echo "  clean         - Full cleanup (cache, build, venv, test artifacts, tokenlearn data)."
	@echo ""
	@echo "Help:"
	@echo "  help          - Display this help message."