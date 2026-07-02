from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, List, Annotated
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
import operator
import ipaddress
import os
from urllib.parse import urlparse
from firecrawl import FirecrawlApp
from langchain_core.tools import tool
from tavily import TavilyClient
from pydantic import BaseModel, Field
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
)

@tool
def search(query: str) -> list[dict]:
    """This tool searches the web for the given query and returns a list of results."""
    client = TavilyClient()
    response = client.search(
        query=query,
        search_depth="advanced",      
        include_raw_content=True,   
        max_results=1,               
    )
    results = []
    for r in response.get("results", []):
        results.append({
            "url": r.get("url"),
            "title": r.get("title"),
            "content": r.get("raw_content") or r.get("content"),
            "score": r.get("score"),
        })
 
    return results

firecrawl = FirecrawlApp(api_key=os.environ["FIRECRAWL_API_KEY"])

# Basic SSRF guardrail — block obviously internal/private targets before fetching.
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}


def _is_url_safe(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.hostname or ""
    if host in _BLOCKED_HOSTS:
        return False
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            return False
    except ValueError:
        pass  # host is a domain name, not a raw IP — fine
    return True


@tool
def fetch_url(url: str) -> dict:
    """
    Fetch and extract the content of a specific webpage.
    Use this when you already have a URL (from search results, or the user
    gave one directly) and need its actual content/structure — e.g. to clone
    a site's layout, not just summarize its text.

    Returns: { markdown, html, title, error }
    """
    if not _is_url_safe(url):
        return {"error": f"Blocked or invalid URL: {url}"}

    try:
        result = firecrawl.scrape_url(
            url,
            params={
                "formats": ["markdown", "html"],   # html preserves structure for cloning
                "onlyMainContent": False,           # keep nav/footer — part of what makes it "the site"
                "waitFor": 2000,                    # ms, lets JS-heavy (React/Vue) sites render
                "timeout": 15000,                   # ms, don't let one slow site hang the pipeline
            },
        )
    except Exception as e:
        return {"error": f"Fetch failed for {url}: {e}"}

    return {
        "markdown": result.get("markdown", ""),
        "html": result.get("html", ""),
        "title": result.get("metadata", {}).get("title", ""),
        "error": None,
    }

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    research_context: str

RESEARCH_SYSTEM_PROMPT = """You are the Research agent for a project-generation pipeline.
Your ONLY job is to gather external information needed before planning a project.
 
You have two tools:
- search(query): use when you don't yet have a specific URL.
- fetch_url(url): use when you have a specific URL and need its actual content/structure.
 
Rules:
- If the user's prompt doesn't reference any external site/product, do NOT call any tool —
  just respond that no research is needed.
- If a URL is given directly, prefer fetch_url over searching for it.
- Once you have enough content to describe the target site's structure and purpose,
  STOP calling tools and summarize what you found.
- Be efficient: don't call more tools than necessary.
"""

model_with_tools = model.bind_tools([search, fetch_url])
def research_node(state: State) -> dict:
    messages = state["messages"]
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + messages
 
    response = model_with_tools.invoke(messages)
    state["messages"] = [response]
    return {
        "messages": state["messages"],
        "research_context": response.content,
    }

tool_node = ToolNode([search, fetch_url])

graph = StateGraph(State)
graph.add_node('research', research_node)
graph.add_node('tools', tool_node)

graph.add_edge(START, 'research')
graph.add_conditional_edges('research', tools_condition)
graph.add_edge('tools','research')

research = graph.compile()
