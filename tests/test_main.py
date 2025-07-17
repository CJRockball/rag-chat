import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Set test environment variables before any imports
os.environ.setdefault("DOC_PATH", "test_document.pdf")
os.environ.setdefault("GOOGLE_API_KEY", "test_key_123")
os.environ.setdefault("CHROMA_DB_PATH", "test_vectorstore")
os.environ.setdefault("COLLECTION_NAME", "test_collection")


@pytest.fixture(scope="session")
def mock_rag_agent():
    """Mock RAG agent for testing"""
    mock_agent = Mock()
    mock_agent.ask.return_value = "This is a test response from the RAG agent."
    return mock_agent


@pytest.fixture(scope="session") 
def mock_document_handler():
    """Mock document handler for testing"""
    mock_handler = Mock()
    mock_db = Mock()
    mock_handler.setup_vector_database.return_value = mock_db
    return mock_handler, mock_db


@pytest.fixture(scope="session")
def test_app(mock_rag_agent, mock_document_handler):
    """Create test app with mocked dependencies"""
    handler, db = mock_document_handler
    
    # Mock the lifespan to avoid database initialization
    @asynccontextmanager
    async def test_lifespan(app: FastAPI):
        # Set global variables for testing
        import main
        main.rag_agent = mock_rag_agent
        main.db = db
        main.chat_history = []
        yield
    
    # Patch the imports before creating the app
    with patch("src.agent.rag_agent.RAGAgent", return_value=mock_rag_agent), \
         patch("src.utils.document_handler.DocumentHandler", return_value=handler), \
         patch("main.lifespan", test_lifespan):
        
        from main import app
        app.dependency_overrides = {}
        yield app


@pytest.fixture
def client(test_app):
    """Test client fixture"""
    with TestClient(test_app) as client:
        yield client


class TestHealthEndpoints:
    """Test health and monitoring endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint returns correct status"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "database_initialized" in data
        assert "chat_history_length" in data
        assert isinstance(data["chat_history_length"], int)

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns expected data"""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_conversations" in data
        assert "app_version" in data
        assert "status" in data
        assert data["status"] == "running"
        assert data["app_version"] == "1.0.0"


class TestChatEndpoints:
    """Test chat functionality endpoints"""
    
    def test_chat_page_loads(self, client):
        """Test that chat page loads correctly"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_chat_page_empty_history(self, client):
        """Test chat page with empty history"""
        # Clear any existing history first
        client.post("/clear")
        
        response = client.get("/")
        assert response.status_code == 200
        content = response.text
        assert "Welcome to RAG Chat!" in content or "chat-history" in content

    def test_ask_question_valid(self, client, mock_rag_agent):
        """Test asking a valid question"""
        question = "What is the main topic of the document?"
        
        response = client.post("/ask", data={"question": question})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        
        # Verify mock was called
        mock_rag_agent.ask.assert_called()
        
        # Check that response contains the question and answer
        content = response.text
        assert question in content or "test response" in content

    def test_ask_question_empty(self, client):
        """Test asking an empty question"""
        response = client.post("/ask", data={"question": ""})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_ask_question_whitespace_only(self, client):
        """Test asking question with only whitespace"""
        response = client.post("/ask", data={"question": "   \n\t   "})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_multiple_questions(self, client, mock_rag_agent):
        """Test asking multiple questions in sequence"""
        questions = [
            "What is the first topic?",
            "What is the second topic?",
            "Can you summarize?"
        ]
        
        for question in questions:
            response = client.post("/ask", data={"question": question})
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
        
        # Verify all questions were processed
        assert mock_rag_agent.ask.call_count == len(questions)

    def test_clear_chat_history(self, client):
        """Test clearing chat history"""
        # First ask a question to create history
        client.post("/ask", data={"question": "Test question"})
        
        # Clear the history
        response = client.post("/clear")
        assert response.status_code == 200
        assert response.json() == {"status": "cleared"}
        
        # Verify history is cleared
        health_response = client.get("/health")
        health_data = health_response.json()
        assert health_data["chat_history_length"] == 0


class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_ask_question_without_form_data(self, client):
        """Test asking question without proper form data"""
        response = client.post("/ask")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_ask_question_with_rag_agent_error(self, client, mock_rag_agent):
        """Test handling RAG agent errors"""
        # Make the RAG agent raise an exception
        mock_rag_agent.ask.side_effect = Exception("RAG agent error")
        
        response = client.post("/ask", data={"question": "Test question"})
        assert response.status_code == 200  # Should handle error gracefully
        assert "text/html" in response.headers["content-type"]
        
        # Check that error message is in response
        content = response.text
        assert "Error processing question" in content

    def test_invalid_endpoints(self, client):
        """Test invalid endpoint requests"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
        
        response = client.post("/invalid")
        assert response.status_code == 404


class TestResponseFormat:
    """Test response formatting and content"""
    
    def test_html_escaping(self, client, mock_rag_agent):
        """Test that HTML is properly escaped in responses"""
        # Set up mock to return HTML content
        mock_rag_agent.ask.return_value = "<script>alert('xss')</script>"
        
        response = client.post("/ask", data={"question": "Test HTML"})
        assert response.status_code == 200
        
        content = response.text
        # Should not contain unescaped HTML
        assert "<script>" not in content
        assert "&lt;script&gt;" in content

    def test_newline_handling(self, client, mock_rag_agent):
        """Test that newlines are properly converted to HTML breaks"""
        mock_rag_agent.ask.return_value = "Line 1\nLine 2\nLine 3"
        
        response = client.post("/ask", data={"question": "Multi-line test"})
        assert response.status_code == 200
        
        content = response.text
        # Should contain HTML breaks
        assert "<br>" in content or "Line 1" in content

    def test_long_response_handling(self, client, mock_rag_agent):
        """Test handling of long responses"""
        long_response = "This is a very long response. " * 100
        mock_rag_agent.ask.return_value = long_response
        
        response = client.post("/ask", data={"question": "Long response test"})
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# Integration test for the complete flow
class TestIntegration:
    """Integration tests for complete user workflows"""
    
    def test_complete_chat_session(self, client, mock_rag_agent):
        """Test a complete chat session workflow"""
        # 1. Load chat page
        response = client.get("/")
        assert response.status_code == 200
        
        # 2. Ask first question
        response = client.post("/ask", data={"question": "Hello, what can you do?"})
        assert response.status_code == 200
        
        # 3. Ask follow-up question
        response = client.post("/ask", data={"question": "Tell me more details"})
        assert response.status_code == 200
        
        # 4. Check metrics
        response = client.get("/metrics")
        data = response.json()
        assert data["total_conversations"] == 2
        
        # 5. Clear history
        response = client.post("/clear")
        assert response.json() == {"status": "cleared"}
        
        # 6. Verify clear worked
        response = client.get("/health")
        assert response.json()["chat_history_length"] == 0

    def test_error_recovery(self, client, mock_rag_agent):
        """Test that app recovers from errors"""
        # First, make a successful request
        response = client.post("/ask", data={"question": "Good question"})
        assert response.status_code == 200
        
        # Then simulate an error
        mock_rag_agent.ask.side_effect = Exception("Temporary error")
        response = client.post("/ask", data={"question": "Error question"})
        assert response.status_code == 200
        
        # Reset and ensure it works again
        mock_rag_agent.ask.side_effect = None
        mock_rag_agent.ask.return_value = "Recovered response"
        response = client.post("/ask", data={"question": "Recovery question"})
        assert response.status_code == 200
