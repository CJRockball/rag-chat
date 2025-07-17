import pytest
import os
from unittest.mock import patch


class TestEnvironmentConfiguration:
    """Test environment variable configuration"""
    
    def test_default_configuration(self):
        """Test default configuration values"""
        # Clear environment variables
        env_vars = ["CHROMA_DB_PATH", "COLLECTION_NAME", "DOC_PATH"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]
        
        # Import main to check defaults
        import main
        
        # Check that defaults are set correctly
        assert main.CHROMA_DB_PATH == "src/utils/vectorstore/db_chroma"
        assert main.COLLECTION_NAME == "v_db"
        assert main.DOC_PATH == "example.pdf"

    def test_custom_configuration(self):
        """Test custom configuration from environment"""
        test_config = {
            "CHROMA_DB_PATH": "custom/path/db",
            "COLLECTION_NAME": "custom_collection",
            "DOC_PATH": "custom_document.pdf"
        }
        
        with patch.dict(os.environ, test_config):
            # Re-import to get updated config
            import importlib
            import main
            importlib.reload(main)
            
            assert main.CHROMA_DB_PATH == "custom/path/db"
            assert main.COLLECTION_NAME == "custom_collection"
            assert main.DOC_PATH == "custom_document.pdf"

    def test_google_api_key_requirement(self):
        """Test that Google API key is required"""
        # This test would need to be adjusted based on actual error handling
        # in the RAG agent initialization
        pass
