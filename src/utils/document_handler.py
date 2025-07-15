import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentHandler:
    """Handles document processing and vector database operations"""

    def __init__(self, model_name: str = "models/text-embedding-004"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_name)

    def setup_vector_database(
        self, doc_path: str, db_path: str, collection_name: str
    ) -> Chroma:
        """Set up vector database with persistence check"""
        if os.path.exists(db_path):
            print("Database exists - connecting to existing database")
            db = Chroma(
                collection_name=collection_name,
                persist_directory=db_path,
                embedding_function=self.embeddings,
            )
        else:
            print("Database does not exist - creating new database")
            db = self._create_new_database(doc_path, db_path, collection_name)
            print("Database created and populated successfully")

        return db

    def _create_new_database(
        self, doc_path: str, db_path: str, collection_name: str
    ) -> Chroma:
        """Create a new vector database from PDF document"""
        # Load and process PDF
        loader = PyPDFLoader(doc_path)
        pages = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)

        # Create vector store from documents
        db = Chroma.from_documents(
            chunks,
            self.embeddings,
            persist_directory=db_path,
            collection_name=collection_name,
        )

        return db

    def test_database(
        self, db: Chroma, query: str = "what is the main deliverable?"
    ) -> None:
        """Test if the vector database is working"""
        docs = db.similarity_search(query)
        print(f"Found {len(docs)} relevant documents")
        if docs:
            print(f"First document preview: {docs[0].page_content[:200]}...")

    def add_document(self, db: Chroma, doc_path: str) -> None:
        """Add a new document to existing database"""
        loader = PyPDFLoader(doc_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunks = text_splitter.split_documents(pages)

        db.add_documents(chunks)
        print(f"Added {len(chunks)} chunks from {doc_path}")
