import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import os

# Set test environment variables before importing main
os.environ.setdefault("DOC_PATH", "test_document.pdf")
os.environ.setdefault("GOOGLE_API_KEY", "test_key_123")


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies before main is imported"""
    with patch("src.agent.rag_agent.RAGAgent") as mock_rag_agent, patch(
        "src.utils.document_handler.DocumentHandler"
    ) as mock_doc_handler:

        # Setup mock RAG agent
        mock_agent_instance = Mock()
        mock_agent_instance.ask.return_value = "Mocked response"
        mock_rag_agent.return_value = mock_agent_instance

        # Setup mock document handler
        mock_db = Mock()
        mock_handler_instance = Mock()
        mock_handler_instance.setup_vector_database.return_value = mock_db
        mock_doc_handler.return_value = mock_handler_instance

        yield {
            "rag_agent": mock_rag_agent,
            "doc_handler": mock_doc_handler,
            "db": mock_db,
            "agent_instance": mock_agent_instance,
        }


@pytest.fixture
def client(mock_dependencies):
    """Test client fixture that imports main after mocking"""
    # Import main AFTER mocking to ensure mocked classes are used
    from main import app

    with TestClient(app) as client:
        yield client


def test_health_check(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_metrics(client):
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_conversations" in data
    assert "app_version" in data
    assert "status" in data


def test_chat_page(client):
    """Test chat page rendering"""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_clear_chat(client):
    """Test clearing chat history"""
    response = client.post("/clear")
    assert response.status_code == 200
    assert response.json() == {"status": "cleared"}


def test_ask_question(client, mock_dependencies):
    """Test asking a question - this should now work"""
    response = client.post("/ask", data={"question": "Test question"})

    # Verify response
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Verify the mock was called
    mock_dependencies["agent_instance"].ask.assert_called_once()

    # Verify the call arguments
    call_args = mock_dependencies["agent_instance"].ask.call_args
    # First argument should be the question
    assert call_args[0][0] == "Test question"
    assert (
        call_args[0][1] == mock_dependencies["db"]
    )  # Second argument should be the database
