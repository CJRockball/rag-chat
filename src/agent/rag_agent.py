import os
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import START, END, StateGraph
from langchain_core.rate_limiters import InMemoryRateLimiter


class State(TypedDict):
    """State definition for the RAG agent"""

    question: str
    context: List[Document]
    answer: str
    db: Chroma


class RAGAgent:
    """LangGraph-based RAG agent for question answering"""

    def __init__(self):
        self.llm = self._initialize_llm()
        self.graph = self._build_graph()

    def _initialize_llm(self) -> ChatGoogleGenerativeAI:
        """Initialize the language model with rate limiting"""
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found \
                in environment variables"
            )

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=0.1,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.1,
            rate_limiter=rate_limiter
        )

    def _retrieve(self, state: State) -> dict:
        """Retrieve relevant documents from vector database"""
        docs = state["db"].similarity_search(query=state["question"], k=3)
        return {"context": docs}

    def _generate(self, state: State) -> dict:
        """Generate answer based on retrieved context"""
        context_text = "\n".join([d.page_content for d in state["context"]])
        system_prompt = ("You are an assistant. Use the context below \
                         to answer the question.\n\n" f"Context:\n{context_text}")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["question"]),
        ]

        response = self.llm.invoke(messages)
        return {"answer": response}

    def _build_graph(self):
        """Build the LangGraph workflow"""
        builder = StateGraph(State)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("generate", self._generate)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "generate")
        builder.add_edge("generate", END)
        graph = builder.compile()
        return graph

    def ask(self, question: str, db: Chroma) -> str:
        """Ask a question and get an answer from the RAG agent"""
        for event in self.graph.stream(
            {"question": question, "db": db}, stream_mode="values"
        ):
            if event.get("answer"):
                ans = event["answer"]
                return ans.content if hasattr(ans, "content") else ans
        return "No answer generated."
