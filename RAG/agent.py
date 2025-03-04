from __future__ import annotations as _annotations

from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
from typing import List, Any

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel  # Importing Pydantic AI's GeminiModel for chat

import chromadb

# Load environment variables from .env file
load_dotenv()

# Initialize ChromaDB persistent client and get the "site_pages" collection
chroma_client = chromadb.PersistentClient(path=r"C:\Users\frank\chroma_db")
try:
    collection = chroma_client.get_collection("site_pages")
except Exception:
    collection = chroma_client.create_collection("site_pages")

@dataclass
class FASTAPIDeps:
    chroma_collection: Any

# Instantiate Pydantic AI's GeminiModel for chat functionality using the provided LLM_MODEL
llm_model_name = os.getenv('LLM_MODEL', 'gemini-2.0-flash')
model = GeminiModel(model_name=llm_model_name)

logfire.configure(send_to_logfire='if-token-present')

async def get_embedding(text: str) -> List[float]:
    """
    Get an embedding for the text using Google's Gemini API embed_content method.
    This uses the 'text-embedding-004' model.
    """
    from google import genai
    # Optionally, you can import types: from google.genai import types
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    result = await asyncio.to_thread(
        client.models.embed_content,
        model="text-embedding-004",
        contents=text
    )
    return result.embeddings

# Create the Pydantic AI agent with the Gemini model (for chat) and updated dependencies
fastapi_expert = Agent(
    model=model,
    system_prompt="""
You are an expert at Pydantic AI - a Python AI agent framework that you have access to all the documentation,
including examples, an API reference, and other resources to help you build Pydantic AI agents.

Your only job is to assist with this and you don't answer other questions besides describing what you are able to do.

Don't ask the user before taking an action, just do it. Always make sure you look at the documentation with the provided tools before answering the user's question unless you have already.

When you first look at the documentation, always start with RAG.
Then also always check the list of available documentation pages and retrieve the content of page(s) if it'll help.

Always let the user know when you didn't find the answer in the documentation or the right URL - be honest.
""",
    deps_type=FASTAPIDeps,
    retries=2
)

@fastapi_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[FASTAPIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query using RAG.
    
    Args:
        ctx: The context including the ChromaDB collection.
        user_query: The user's question or query.
        
    Returns:
        A formatted string containing the top 5 most relevant documentation chunks.
    """
    try:
        # Get the embedding for the query using Google's Gemini API embed_content method
        query_embedding = await get_embedding(user_query)
        
        # Query ChromaDB for relevant documents
        result = await asyncio.to_thread(
            ctx.deps.chroma_collection.query,
            query_embeddings=[query_embedding],
            n_results=5,
            where={"source": "fastapi_docs"}
        )
        
        if not result or not result.get("documents", []) or not result["documents"][0]:
            return "No relevant documentation found."
            
        # Format the results (ChromaDB returns lists of lists)
        formatted_chunks = []
        for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
            chunk_text = f"""
# {meta.get('title', 'No Title')}

{doc}
"""
            formatted_chunks.append(chunk_text.strip())
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

@fastapi_expert.tool
async def list_documentation_pages(ctx: RunContext[FASTAPIDeps]) -> List[str]:
    """
    Retrieve a list of all available FastAPI documentation pages.
    
    Returns:
        List[str]: Unique URLs for all documentation pages.
    """
    try:
        docs = await asyncio.to_thread(
            ctx.deps.chroma_collection.get,
            where={"source": "fastapi_docs"}
        )
        
        if not docs or not docs.get("metadatas"):
            return []
            
        # Extract unique URLs from metadata
        urls = sorted(set(meta.get('url', '') for meta in docs["metadatas"] if meta.get('url')))
        return urls
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@fastapi_expert.tool
async def get_page_content(ctx: RunContext[FASTAPIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page by combining its chunks.
    
    Args:
        ctx: The context including the ChromaDB collection.
        url: The URL of the page to retrieve.
        
    Returns:
        The complete page content with all chunks combined in order.
    """
    try:
        docs = await asyncio.to_thread(
            ctx.deps.chroma_collection.get,
            where={"url": url, "source": "fastapi_docs"}
        )
        
        if not docs or not docs.get("documents"):
            return f"No content found for URL: {url}"
            
        # Pair metadata and document text, then sort by chunk_number
        paired = list(zip(docs["metadatas"], docs["documents"]))
        paired.sort(key=lambda x: x[0].get("chunk_number", 0))
        
        page_title = paired[0][0].get("title", "Untitled").split(" - ")[0]
        formatted_content = [f"# {page_title}\n"]
        
        for meta, doc in paired:
            formatted_content.append(doc)
            
        return "\n\n".join(formatted_content)
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
