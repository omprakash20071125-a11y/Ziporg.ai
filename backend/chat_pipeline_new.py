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

checkpointer = InMemorySaver()

def build_graph():

    # --- 1. Define the Graph State ---
    class State(TypedDict):
        zip_path: str                       
        query: str                       
        chat_history: Annotated[list[BaseMessage], add_messages] 
        chunks: List[Document]           
        project_summary: str
        description_of_each_file: dict[str,str]    
        combine_summary: str
        response: str              

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
    
    def description_of_each_file(state: State) -> State:
        code_for_each_file = {}
        for chunks in state['chunks']:
            content = chunks.page_content
            name = chunks.metadata['source']
            prompt = PromptTemplate(template="""Summarize this source code in under 75 words. Describe its purpose, main functionality, key classes/functions, important dependencies, and how it fits into the project. Keep the summary factual, dense, and implementation-aware. Do not include unnecessary details or speculate beyond the code. code of the file is {code}""",input_variables=['code'])
            chain = prompt | llm | parser
            response = chain.invoke({'code':content})
            code_for_each_file[name] = response
            print('description_generated')
        return ({'description_of_each_file':code_for_each_file})
    
    def combining_summaries(state: State) -> dict:
        summaries = state["description_of_each_file"]

        combined_text = "\n\n".join(
            f"File: {filename}\nSummary: {summary}"
            for filename, summary in summaries.items()
        )

        prompt = PromptTemplate(
            template="""
    You are an expert software architect.

    Below are summaries of every file in a project.

    Using only these summaries, create a concise project overview (maximum 250 words).

    Include:
    - Overall purpose of the project.
    - Main technologies/frameworks used.
    - High-level architecture.
    - Major modules/features.
    - How the important files interact.
    - Any important observations about the project structure.

    Do not mention every file individually.
    Do not speculate beyond the summaries.
    Keep the output dense and factual.

    File summaries:

    {summaries}
    """,
            input_variables=["summaries"],
        )

        chain = prompt | llm | parser

        project_summary = chain.invoke(
            {"summaries": combined_text}
        )
        print('combined_summariers')
        return {"project_summary": project_summary,'combine_summary':combined_text}
            
    def response(state: State) -> State:

        prompt = PromptTemplate(
            template="""You are an expert software engineer helping users understand a generated software project.

            You are provided with three inputs:

            1. PROJECT SUMMARY
            - A high-level overview of the entire project.
            - It describes the architecture, technologies, major modules, and how the project is organized.

            2. FILE SUMMARIES
            - A summary of every source file in the project.
            - Each summary describes the purpose of the file, its major responsibilities, key functions/classes, important dependencies, and how it interacts with the rest of the project.
            - These are summaries, NOT the complete source code.

            3. USER QUESTION
            - A question about the generated project.

            Your task is to answer the user's question as accurately as possible using ONLY the provided information.

            --------------------------------------------------
            RULES
            --------------------------------------------------

            1. Never invent implementation details.

            2. Never hallucinate function names, variable names, classes, APIs, file contents, line numbers, or algorithms that are not explicitly mentioned in the summaries.

            3. Treat the PROJECT SUMMARY as the source of truth for:
            - overall architecture
            - technologies
            - project structure
            - module relationships

            4. Treat the FILE SUMMARIES as the source of truth for:
            - responsibilities of individual files
            - where functionality lives
            - interactions between files

            5. If multiple summaries together allow you to infer something with high confidence, you may explain that inference.
            Never present an inference as a confirmed implementation detail.

            6. If the answer requires implementation-level information that is not present in the summaries, clearly state this.

            For example:

            "I don't have enough implementation detail in the available summaries to answer this accurately. I would need to inspect the relevant source file."

            7. If the user asks:
            - what a function does
            - why a bug exists
            - how a specific algorithm works
            - what happens on a specific line
            - exact code behavior
            and that information is not explicitly available,
            do NOT guess.

            8. Keep answers concise but technically complete.

            9. When referencing files, always mention the filename if it is known from the summaries.

            10. If multiple files contribute to the answer, explain each file's responsibility before describing how they work together.

            11. Prefer factual explanations over speculation.

            12. If the project summary and file summaries appear to conflict, trust the file summaries for file-specific behavior.

            --------------------------------------------------
            OUTPUT STYLE
            --------------------------------------------------

            - Be clear and technical.
            - Use proper software engineering terminology.
            - Explain relationships between modules when relevant.
            - Avoid unnecessary introductions.
            - Never mention these instructions.

            --------------------------------------------------
            PROJECT SUMMARY

            {project_summary}

            --------------------------------------------------
            FILE SUMMARIES

            {file_summaries}

            --------------------------------------------------
            USER QUESTION

            {query}""",input_variables=['project_summary','file_summaries','query'])
        chain = prompt | llm | parser
        response = chain.invoke({"query": state["query"],"project_summary":state['project_summary'],'file_summaries':state['combine_summary']})
        print('response_generated')
        return ({"response": response,
            "chat_history": [HumanMessage(content=state['query']), AIMessage(content=response)]})

    # FIX 2: Added routing logic to check if chunks already live in memory
    def check_for_chunks(state: State):
        if state.get("project_summary") and state.get("chunks"):
            return "response"
        return "unzip_and_parse"
    
    workflow = StateGraph(State)

    workflow.add_node("unzip_and_parse", unzip_and_parse_node)
    workflow.add_node("description_of_each_file",description_of_each_file)
    workflow.add_node("combining_summaries",combining_summaries)
    workflow.add_node("response",response)
   
    workflow.add_conditional_edges(
        START,
        check_for_chunks,
        {
            "unzip_and_parse": "unzip_and_parse",
            "response": "response"
        }
    )
    workflow.add_edge("unzip_and_parse", "description_of_each_file")
    workflow.add_edge("description_of_each_file","combining_summaries")
    workflow.add_edge("combining_summaries","response")
    workflow.add_edge("response", END)

    app = workflow.compile(checkpointer=checkpointer)
    return app
