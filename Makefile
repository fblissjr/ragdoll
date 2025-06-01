# Makefile for the RAGdoll Project
# Defines common development, testing, and application execution tasks,
# including helper targets for preparing custom embedding models using Tokenlearn.

# --- Variables ---
# Attempt to use 'python3' by default; can be overridden.
PYTHON_INTERPRETER ?= python3
# Use 'uv' for environment and package management. Assumes 'uv' is installed.
# If uv is directly on PATH:
UV := uv
# If uv is installed via pip in the Python interpreter being used:
# UV := $(PYTHON_INTERPRETER) -m uv

# Default virtual environment directory name
VENV_DIR ?= .venv

# --- Environment Setup ---
.PHONY: venv install-dev install help

# Creates a Python virtual environment using uv.
venv:
	@echo ">>> Creating virtual environment in $(VENV_DIR) using uv..."
	$(UV) venv $(VENV_DIR) --python $(PYTHON_INTERPRETER)
	@echo ">>> Virtual environment created. Activate with: source $(VENV_DIR)/bin/activate"
	@echo ">>> To install dependencies, run: make install-dev"

# Installs all dependencies, including development extras like pytest, ruff, tokenlearn.
install-dev: venv
	@echo ">>> Installing all dependencies (including development extras) into $(VENV_DIR) using uv..."
	$(VENV_DIR)/bin/$(UV) sync --all-extras
	@echo ">>> Installing pre-commit hooks..."
	$(VENV_DIR)/bin/$(UV) run pre-commit install
	@echo ">>> Dependencies and pre-commit hooks installed."

# Installs only production/main dependencies.
install: venv
	@echo ">>> Installing production dependencies into $(VENV_DIR) using uv..."
	$(VENV_DIR)/bin/$(UV) sync # Assumes main dependencies are listed under [project.dependencies]
	@echo ">>> Production dependencies installed."

# --- Code Quality & Formatting ---
.PHONY: lint format check-types fix-code

# Runs linters (Ruff) to check for code style and potential errors.
lint:
	@echo ">>> Running linters (Ruff check) on src and tests..."
	$(VENV_DIR)/bin/$(UV) run ruff check src tests

# Formats code using Ruff formatter.
format:
	@echo ">>> Formatting code (Ruff format) for src and tests..."
	$(VENV_DIR)/bin/$(UV) run ruff format src tests

# Runs static type checks using MyPy.
check-types:
	@echo ">>> Running static type checks (MyPy) on src..."
	$(VENV_DIR)/bin/$(UV) run mypy src

# Runs pre-commit hooks on all files to automatically fix issues.
fix-code:
	@echo ">>> Running pre-commit hooks to fix files..."
	$(VENV_DIR)/bin/$(UV) run pre-commit run --all-files

# --- Testing ---
.PHONY: test test-cov

# Runs automated tests using pytest.
test:
	@echo ">>> Running tests (pytest)..."
	$(VENV_DIR)/bin/$(UV) run pytest tests/

# Runs tests with code coverage reporting.
test-cov:
	@echo ">>> Running tests with coverage (pytest-cov)..."
	$(VENV_DIR)/bin/$(UV) run pytest --cov=src --cov-report=term-missing --cov-report=html tests/
	@echo ">>> Coverage report generated in htmlcov/ directory."

# --- RAGdoll Application Specific Tasks ---
.PHONY: run-api process-data-example run-cli

# Starts the RAGdoll FastAPI service using Uvicorn.
run-api:
	@echo ">>> Starting RAGdoll API service (Uvicorn)..."
	# Assumes src/main_api.py contains the FastAPI app instance named 'app'.
	$(VENV_DIR)/bin/$(UV) run uvicorn src.main_api:app --reload --host 0.0.0.0 --port 8001

# Example target for running the data processing CLI with common settings.
# Users should customize --docs-folder and other parameters as needed.
process-data-example:
	@echo ">>> Running RAGdoll data processing pipeline (example configuration)..."
	$(VENV_DIR)/bin/$(UV) run python -m src.process_data_cli \
		--docs-folder ./sample_docs_for_processing \
		--vector-data-dir ./vector_store_output \
		--overwrite \
		--chunker-type chonkie_sdpm \
		--embedding-model-name "sentence-transformers/all-MiniLM-L6-v2" \
		--enable-classification \
		--prepare-viz-data \
		--verbose
	@echo ">>> Example data processing finished."

# Starts the RAGdoll interactive Command Line Interface.
run-cli:
	@echo ">>> Starting RAGdoll Interactive CLI..."
	$(VENV_DIR)/bin/$(UV) run python -m src.rag_cli

# --- Tokenlearn Model Preparation (Helper Targets) ---
# These targets help run Tokenlearn scripts to prepare custom embedding models.
# Ensure 'tokenlearn' package is installed in the environment (e.g., via 'make install-dev'
# if 'tokenlearn' is listed in RAGdoll's [project.optional-dependencies.dev]).
# Users should customize TOKENLEARN_* variables below or override them via the command line.

TOKENLEARN_BASE_MODEL ?= "baai/bge-base-en-v1.5" # Teacher model for Tokenlearn
TOKENLEARN_CORPUS_HF_PATH ?= "allenai/c4"        # Hugging Face dataset path for featurization
TOKENLEARN_CORPUS_HF_NAME ?= "en"                # Dataset configuration/subset name
TOKENLEARN_CORPUS_HF_SPLIT ?= "train"            # Dataset split (e.g., train, validation)
TOKENLEARN_CORPUS_TEXT_KEY ?= "text"             # Key in the dataset containing the text

TOKENLEARN_MAX_MEANS ?= 500000                   # Max passage means for featurization (adjust for time/corpus size)
TOKENLEARN_FEATURIZE_BATCH_SIZE ?= 64            # Batch size for featurization encoding

TOKENLEARN_PCA_DIMS ?= 256                       # Target dimension after PCA in Tokenlearn training
TOKENLEARN_VOCAB_SIZE ?= 32000                   # Custom vocabulary size for Tokenlearn (0 to disable)
TOKENLEARN_TRAIN_DEVICE ?= "cpu"                 # Device for Tokenlearn training ("cpu", "cuda:0", "mps")

# Output directories relative to RAGdoll project root (CURDIR)
RAGDOLL_TOKENLEARN_FEATURES_DIR := $(CURDIR)/data_for_tokenlearn/features_$(subst /,-,$(TOKENLEARN_BASE_MODEL))
RAGDOLL_TOKENLEARN_MODEL_DIR := $(CURDIR)/custom_models/tokenlearned_$(subst /,-,$(TOKENLEARN_BASE_MODEL))_pca$(TOKENLEARN_PCA_DIMS)_vocab$(TOKENLEARN_VOCAB_SIZE)

.PHONY: tokenlearn-prepare tokenlearn-custom-train tokenlearn-custom-clean

# Runs Tokenlearn's featurization script.
tokenlearn-prepare:
	@echo ">>> Preparing features for custom Tokenlearn model using '$(TOKENLEARN_BASE_MODEL)'..."
	@echo "    Corpus: HF:'$(TOKENLEARN_CORPUS_HF_PATH)' (name:'$(TOKENLEARN_CORPUS_HF_NAME)', split:'$(TOKENLEARN_CORPUS_HF_SPLIT)', key:'$(TOKENLEARN_CORPUS_TEXT_KEY)')"
	@echo "    Output features to: $(RAGDOLL_TOKENLEARN_FEATURES_DIR)"
	@mkdir -p $(RAGDOLL_TOKENLEARN_FEATURES_DIR)
	$(VENV_DIR)/bin/$(UV) run python -m tokenlearn.featurize \
		--model-name "$(TOKENLEARN_BASE_MODEL)" \
		--output-dir "$(RAGDOLL_TOKENLEARN_FEATURES_DIR)" \
		--dataset-path "$(TOKENLEARN_CORPUS_HF_PATH)" \
		--dataset-name "$(TOKENLEARN_CORPUS_HF_NAME)" \
		--dataset-split "$(TOKENLEARN_CORPUS_HF_SPLIT)" \
		--key "$(TOKENLEARN_CORPUS_TEXT_KEY)" \
		--max-means $(TOKENLEARN_MAX_MEANS) \
		--batch-size $(TOKENLEARN_FEATURIZE_BATCH_SIZE)
	@echo ">>> Tokenlearn featurization complete."

# Runs Tokenlearn's training script using previously generated features.
tokenlearn-custom-train: $(RAGDOLL_TOKENLEARN_FEATURES_DIR) # Ensures features exist (or prints message from dummy target)
	@echo ">>> Training custom embedding model with Tokenlearn..."
	@echo "    Base model (for tokenizer/initial distill): $(TOKENLEARN_BASE_MODEL)"
	@echo "    Reading features from: $(RAGDOLL_TOKENLEARN_FEATURES_DIR)"
	@echo "    Saving trained model to: $(RAGDOLL_TOKENLEARN_MODEL_DIR)"
	@echo "    PCA Dimensions: $(TOKENLEARN_PCA_DIMS), Custom Vocab Size: $(TOKENLEARN_VOCAB_SIZE)"
	@echo "    Training Device: $(TOKENLEARN_TRAIN_DEVICE)"
	@mkdir -p $(RAGDOLL_TOKENLEARN_MODEL_DIR)
	$(VENV_DIR)/bin/$(UV) run python -m tokenlearn.train \
		--model-name "$(TOKENLEARN_BASE_MODEL)" \
		--data-path "$(RAGDOLL_TOKENLEARN_FEATURES_DIR)" \
		--save-path "$(RAGDOLL_TOKENLEARN_MODEL_DIR)" \
		--pca-dims $(TOKENLEARN_PCA_DIMS) \
		--vocab-size $(TOKENLEARN_VOCAB_SIZE) \
		--device "$(TOKENLEARN_TRAIN_DEVICE)" 
		# Add other tokenlearn.train args like --lr, --patience, --batch_size as needed
	@echo ">>> Tokenlearn training complete. Custom model saved to: $(RAGDOLL_TOKENLEARN_MODEL_DIR)"
	@echo ">>> You can now use this path as '--embedding-model-name' in RAGdoll's 'process-data' or API."

# Dummy target to satisfy dependency and provide a message if features are missing.
$(RAGDOLL_TOKENLEARN_FEATURES_DIR):
	@echo "Tokenlearn features directory '$(RAGDOLL_TOKENLEARN_FEATURES_DIR)' not found."
	@echo "Please run 'make tokenlearn-prepare' first."
	@exit 1


# Cleans up Tokenlearn-generated features and custom models.
tokenlearn-custom-clean:
	@echo ">>> Cleaning Tokenlearn generated features and custom model directories..."
	@rm -rf "$(RAGDOLL_TOKENLEARN_FEATURES_DIR)"
	@rm -rf "$(RAGDOLL_TOKENLEARN_MODEL_DIR)"
	@echo ">>> Tokenlearn custom data cleaned."

# --- General Cleanup Targets ---
.PHONY: clean clean-pyc clean-build

# Removes Python cache files and directories.
clean-pyc:
	@echo ">>> Removing Python cache files (__pycache__, .pyc, .pyo)..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete

# Removes build artifacts.
clean-build:
	@echo ">>> Removing build artifacts (build/, dist/, .eggs/, *.egg-info, *.egg)..."
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.egg" -delete

# Comprehensive cleanup: Python cache, build artifacts, test artifacts, Tokenlearn data, and virtual env.
clean: clean-pyc clean-build tokenlearn-custom-clean
	@echo ">>> Removing test cache (.pytest_cache) and coverage data (htmlcov/, .coverage)..."
	rm -rf .pytest_cache
	rm -rf htmlcov/
	rm -f .coverage
	@echo ">>> Removing virtual environment $(VENV_DIR)..."
	rm -rf $(VENV_DIR)
	@echo ">>> Full project cleanup complete."

# Default target when 'make' is run without arguments.
default: help

# Displays help message with available targets.
help:
	@echo "RAGdoll Project Makefile"
	@echo "--------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Environment & Installation:"
	@echo "  venv          - Create Python virtual environment using uv."
	@echo "  install-dev   - Install all dependencies including development tools (requires venv)."
	@echo "  install       - Install only production dependencies (requires venv)."
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