# tests/conftest.py
import pytest
import tempfile
import shutil
import os
from pathlib import Path

# Dummy content for sample files
SAMPLE_TXT_CONTENT = "This is a sample text file for testing the RAGdoll pipeline."
SAMPLE_MD_CONTENT = """# Markdown Sample

This is a sample markdown document.

- Item 1
- Item 2

Another paragraph.
"""
SAMPLE_JSON_CONTENT = """
{
  "name": "RAGdoll Test",
  "version": "1.0",
  "details": {
    "description": "Testing JSON parsing.",
    "items": ["itemA", "itemB"]
  }
}
"""
SAMPLE_CSV_CONTENT = """id,name,value
1,alpha,100
2,beta,200
3,gamma,300
"""

@pytest.fixture(scope="function") # Use "function" scope for clean state per test
def temp_docs_dir():
    """Creates a temporary directory with sample documents for testing."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ragdoll_test_docs_"))
    sample_files_content = {
        "sample.txt": SAMPLE_TXT_CONTENT,
        "sample.md": SAMPLE_MD_CONTENT,
        "sample.json": SAMPLE_JSON_CONTENT,
        "sample.csv": SAMPLE_CSV_CONTENT,
        # For DOCX and PDF, it's harder to create them programmatically here.
        # For robust testing, you should add actual small sample.docx and sample.pdf
        # files to tests/sample_docs/ and copy them here.
        # For now, we'll skip them to keep this example self-contained,
        # but acknowledge they are important for full testing.
    }
    
    for filename, content in sample_files_content.items():
        with open(temp_dir / filename, "w", encoding="utf-8") as f:
            f.write(content)

    # Placeholder for actual DOCX and PDF - copy them from a predefined location
    # For real tests, ensure these files exist in your test setup.
    # For example, if you have `tests/sample_files/sample.docx`:
    # test_sample_files_dir = Path(__file__).parent / "sample_files"
    # if (test_sample_files_dir / "sample.docx").exists():
    #     shutil.copy(test_sample_files_dir / "sample.docx", temp_dir / "sample.docx")
    # else:
    #     print("Warning: sample.docx not found for testing.")
    # if (test_sample_files_dir / "sample.pdf").exists():
    #     shutil.copy(test_sample_files_dir / "sample.pdf", temp_dir / "sample.pdf")
    # else:
    #     print("Warning: sample.pdf not found for testing.")

    print(f"Created temp_docs_dir: {temp_dir}")
    yield temp_dir # Provide the path to the test

    # Teardown: remove the temporary directory after the test
    print(f"Removing temp_docs_dir: {temp_dir}")
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def temp_output_dir():
    """Creates a temporary directory for pipeline output."""
    temp_dir = Path(tempfile.mkdtemp(prefix="ragdoll_test_output_"))
    print(f"Created temp_output_dir: {temp_dir}")
    yield temp_dir

    # Teardown
    print(f"Removing temp_output_dir: {temp_dir}")
    shutil.rmtree(temp_dir)