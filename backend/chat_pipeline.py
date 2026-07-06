import os
import base64
import io
import zipfile
from typing import TypedDict, List, Annotated

from dotenv import load_dotenv
from langgraph.graph.message import add_messages
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever  
from langchain_google_genai import ChatGoogleGenerativeAI

from langgraph.graph import StateGraph, START, END

import cohere

load_dotenv()

parser = StrOutputParser()

# Reuse one LLM client instead of instantiating a new one per node call
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)
llm_final = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

co = cohere.ClientV2(os.getenv("COHERE_API_KEY"))

checkpointer = InMemorySaver()

def build_graph():

    # --- 1. Define the Graph State ---
    class State(TypedDict):
        zip_path: str                       # incoming base64 zip string
        query: str                          # user's question
        chat_history: Annotated[list[BaseMessage], add_messages]  # prior turns, defaults to []
        chunks: List[Document]              # extracted / split chunks
        hypo_answer: str                    # HyDE hypothetical answer
        response: str                       # final generated answer

    # --- 2. Helpers & Nodes ---
    BINARY_EXTENSIONS = {
        ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".svg",
        ".woff", ".woff2", ".ttf", ".eot",
        ".zip", ".tar", ".gz", ".pdf",
        ".pyc", ".so", ".dll", ".exe", ".bin",
        ".lock",
    }

    SKIP_DIR_MARKERS = ("node_modules/", "__pycache__/", "/.git/", ".git/")

    def unzip_and_parse_node(state: State) -> dict:
        """Node: converts the base64 zip into raw LangChain Document objects (one per file)."""
        b64 = state["zip_path"]
        if b64.startswith("data:"):
            _, b64 = b64.split(",", 1)

        zip_bytes = base64.b64decode(b64)
        documents: List[Document] = []

        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                path = info.filename
                ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""
                if ext in BINARY_EXTENSIONS:
                    continue
                if any(marker in path for marker in SKIP_DIR_MARKERS):
                    continue

                raw = zf.read(info)
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    continue  

                documents.append(Document(page_content=text, metadata={"source": path}))
                print('fething of documents done')

        return {"chunks": documents}

    def chunk_text_node(state: State) -> dict:
        """Node: splits raw per-file documents into retrieval-sized chunks."""
        docs = state["chunks"]
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        print('chunking of docs done')
        return {"chunks": chunks}

    def hypo_answer_generator_node(state: State) -> dict:
        """Node: HyDE — generate a hypothetical answer to improve retrieval quality."""
        hyde_prompt = PromptTemplate(
            template=(
                "Write a hypothetical, highly technical, and detailed paragraph or passage "
                "answering the following question: '{query}'. Provide only the direct factual "
                "text answer as if it were a page excerpt pulled from a textbook."
            ),
            input_variables=["query"],
        )
        chain = hypo_answer_generator = hyde_prompt | llm | parser
        hypo_answer = chain.invoke({"query": state["query"]})
        return {"hypo_answer": hypo_answer}  

    def generate_response_node(state: State) -> dict:
        """Node: ensemble retrieval (FAISS + BM25), Cohere rerank, and final LLM generation."""
        chunks = state["chunks"]
        query = state["query"]
        hypo_answer = state["hypo_answer"]   
        chat_history = state.get("chat_history") or []

        # A. Build ephemeral vector store for this request
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)

        # B. Ensemble retrieval (vector + lexical), queried with the HyDE passage
        vector_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        keyword_retriever = BM25Retriever.from_documents(chunks)
        keyword_retriever.k = 5
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vector_retriever, keyword_retriever],
            weights=[0.5, 0.5],
        )
        retrieved_docs = ensemble_retriever.invoke(hypo_answer)
        content_list = [doc.page_content for doc in retrieved_docs]

        if not content_list:
            return {"response": "I couldn't find anything relevant in this project to answer that."}

        # C. Cohere rerank
        rerank_response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=content_list,
            top_n=min(3, len(content_list)),
        )
        reranked_context = [content_list[r.index] for r in rerank_response.results]
        context_str = "\n\n".join(reranked_context)

        # D. Final LLM generation
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an incredibly intellectual and helpful assistant. Answer the user prompt based strictly on this context:\n\n{context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{query}"),
        ])
        chain = prompt | llm_final | parser
        final_output = chain.invoke({
            "context": context_str,
            "query": query,
            "chat_history": chat_history,  
        })
        print("response generated")
        
        # FIX 1: Return real Message objects so the add_messages reducer can append them cleanly
        return {
            "response": final_output,
            "chat_history": [HumanMessage(content=query), AIMessage(content=final_output)]
        }

    # FIX 2: Added routing logic to check if chunks already live in memory
    def check_for_chunks(state: State):
        if state.get("chunks"):
            return "hypo_answer_generator"
        return "unzip_and_parse"

    # --- 3. Build and Compile the LangGraph Architecture ---
    workflow = StateGraph(State)

    workflow.add_node("unzip_and_parse", unzip_and_parse_node)
    workflow.add_node("chunk_text", chunk_text_node)
    workflow.add_node("hypo_answer_generator", hypo_answer_generator_node)
    workflow.add_node("generate_response", generate_response_node)

    # FIX 3: Replaced standard START edge with conditional edge routing
    workflow.add_conditional_edges(
        START,
        check_for_chunks,
        {
            "unzip_and_parse": "unzip_and_parse",
            "hypo_answer_generator": "hypo_answer_generator"
        }
    )
    workflow.add_edge("unzip_and_parse", "chunk_text")
    workflow.add_edge("chunk_text", "hypo_answer_generator")
    workflow.add_edge("hypo_answer_generator", "generate_response")
    workflow.add_edge("generate_response", END)

    app = workflow.compile(checkpointer=checkpointer)
    return app
