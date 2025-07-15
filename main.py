import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.agent.rag_agent import RAGAgent
from src.utils.document_handler import DocumentHandler
import html
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH",
                                "src/utils/vectorstore/db_chroma")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "v_db")
DOC_PATH = os.environ.get("DOC_PATH", "example.pdf")

# Global variables that will be initialized during lifespan
rag_agent = None
db = None
chat_history = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown events"""
    global rag_agent, db

    # Startup
    print("🚀 Starting up FastAPI application...")

    try:
        # Initialize components
        rag_agent = RAGAgent()
        doc_handler = DocumentHandler()

        # Initialize database
        db = doc_handler.setup_vector_database(
            DOC_PATH, CHROMA_DB_PATH, COLLECTION_NAME
        )
        print("✅ Database initialized successfully")

    except Exception as e:
        print(f"❌ Warning: Could not initialize database: {e}")
        db = None

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
