import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.agent.rag_agent import RAGAgent
from src.utils.document_handler import DocumentHandler
import html
from dotenv import load_dotenv
import logging 
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
#CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH",
#                                "/app/src/utils/vectorstore/db_chroma")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "v_db")
#DOC_PATH = os.environ.get("DOC_PATH", 
#     "/app/documents/Tender_Specs_ECDC2024OP0017_V1.pdf")

# Global variables that will be initialized during lifespan
rag_agent = None
db = None
chat_history = []

def validate_file_paths():
    """Validate that all required files exist and are accessible"""
    doc_path = os.environ.get('DOC_PATH')
    
    if not doc_path:
        raise ValueError("DOC_PATH environment variable is not set")
    
    logger.info(f"Checking PDF file path: {doc_path}")
    
    # Check if file exists
    if not os.path.exists(doc_path):
        # List available files for debugging
        doc_dir = os.path.dirname(doc_path)
        if os.path.exists(doc_dir):
            available_files = os.listdir(doc_dir)
            logger.error(f"PDF file not found at: {doc_path}")
            logger.error(f"Available files in {doc_dir}: {available_files}")
        else:
            logger.error(f"Directory does not exist: {doc_dir}")
        raise FileNotFoundError(f"PDF file not found: {doc_path}")
    
    # Check if file is readable
    if not os.access(doc_path, os.R_OK):
        raise PermissionError(f"Cannot read PDF file: {doc_path}")
    
    logger.info(f"✅ PDF file validated: {doc_path}")
    return doc_path

def ensure_directories():
    """Ensure necessary directories exist"""
    db_path = os.environ.get('CHROMA_DB_PATH', '/src/utils/vectorstore/db_chroma')
    db_dir = Path(db_path).parent
    
    db_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"✅ Database directory ready: {db_dir}")
    return db_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events"""
    global rag_agent, db

    # Startup
    print("🚀 Starting up FastAPI application...")

    try:
        validated_doc_path = validate_file_paths()
        validated_db_path = ensure_directories()
    
        # Initialize components
        rag_agent = RAGAgent()
        doc_handler = DocumentHandler()
    
        # Setup database with validated paths
        db = doc_handler.setup_vector_database(
            validated_doc_path, 
            validated_db_path, 
            COLLECTION_NAME
        )
        logger.info("✅ Database initialized successfully")
    
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        logger.error(f"Current working directory: {os.getcwd()}")
        logger.error(f"Environment variables: DOC_PATH={os.environ.get('DOC_PATH')}")
        logger.error(f"Directory listing: {os.listdir('.')}")
        raise

    yield

    # Shutdown
    print("🔥 Shutting down FastAPI application...")


# Initialize FastAPI app with lifespan
app = FastAPI(title="RAG Chat Application", version="1.0.0", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


def format_message_for_display(message: str) -> str:
    """Format message for proper HTML display"""
    message = html.escape(message)
    message = message.replace("\n", "<br>")
    return message


def get_db():
    """Dependency to get database instance"""
    if db is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    return db


def get_rag_agent():
    """Dependency to get RAG agent instance"""
    if rag_agent is None:
        raise HTTPException(status_code=500,
                            detail="RAG agent not initialized")
    return rag_agent


@app.get("/", response_class=HTMLResponse)
async def chat_page(request: Request):
    """Render the main chat interface"""
    formatted_history = []
    for msg in chat_history:
        formatted_msg = {
            "user": format_message_for_display(msg["user"]),
            "bot": format_message_for_display(msg["bot"]),
        }
        formatted_history.append(formatted_msg)

    return templates.TemplateResponse(
        "chat.html", {"request": request, "chat_history": formatted_history}
    )


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(
    request: Request,
    question: str = Form(...),
    database=Depends(get_db),
    agent=Depends(get_rag_agent),
):
    """Handle user questions and return responses"""
    if not question.strip():
        return templates.TemplateResponse(
            "chat.html", {"request": request, "chat_history": chat_history}
        )

    try:
        # Get answer from RAG agent
        answer = agent.ask(question.strip(), database)

        # Store in chat history
        chat_history.append({"user": question.strip(), "bot": answer})

        # Format for display
        formatted_history = []
        for msg in chat_history:
            formatted_msg = {
                "user": format_message_for_display(msg["user"]),
                "bot": format_message_for_display(msg["bot"]),
            }
            formatted_history.append(formatted_msg)

    except Exception as e:
        error_msg = f"Error processing question: {str(e)}"
        chat_history.append({"user": question.strip(), "bot": error_msg})

        formatted_history = []
        for msg in chat_history:
            formatted_msg = {
                "user": format_message_for_display(msg["user"]),
                "bot": format_message_for_display(msg["bot"]),
            }
            formatted_history.append(formatted_msg)

    return templates.TemplateResponse(
        "chat.html", {"request": request, "chat_history": formatted_history}
    )


@app.post("/clear")
async def clear_chat():
    """Clear chat history"""
    global chat_history
    chat_history = []
    return {"status": "cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database_initialized": db is not None,
        "chat_history_length": len(chat_history),
    }


@app.get("/metrics")
async def metrics():
    """Basic metrics endpoint"""
    return {
        "total_conversations": len(chat_history),
        "app_version": "1.0.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
