import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ.setdefault("DOC_PATH", "test_document.pdf")
os.environ.setdefault("GOOGLE_API_KEY", "test_key_123")
os.environ.setdefault("CHROMA_DB_PATH", "test_vectorstore")
os.environ.setdefault("COLLECTION_NAME", "test_collection")

# Configure pytest
pytest_plugins = []

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment"""
    # Ensure test directories exist
    test_dirs = ["test_vectorstore", "test_documents"]
    for directory in test_dirs:
        Path(directory).mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after tests
    import shutil
    for directory in test_dirs:
        if Path(directory).exists():
            shutil.rmtree(directory)
