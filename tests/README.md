# Tests for RAGdoll

This directory contains automated tests for the RAGdoll project, primarily using `pytest`.

## Running Tests

Ensure you have installed the development dependencies:
```bash
make install-dev 
# or directly: uv pip install pytest pytest-cov
```

To run all tests:

```bash
make test
```

To run tests with coverage:

```bash
make test-cov
```

A detailed HTML coverage report will be generated in the `htmlcov/` directory.

## Structure

* **`conftest.py`**: Contains shared pytest fixtures, such as those for creating temporary directories for test input documents and pipeline outputs.
* **`test_pipeline_processing.py`**: Focuses on testing the data processing pipeline orchestrated by `pipeline_orchestrator.py` and invoked via `process_data_cli.py`. It tests different chunkers, file formats, and pipeline configurations.
* **`sample_files/`**: This subdirectory should contain small, valid sample files (e.g., `.docx`, `.pdf`, `.epub`) that are used by the tests in `conftest.py` to populate temporary document directories. These files are essential for testing the file parsing and conversion logic. **You need to add these sample files manually.**