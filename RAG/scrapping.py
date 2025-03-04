import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

import ollama  # Use Ollama for LLM tasks
import chromadb  # Use ChromaDB for vector storage

# Load environment variables from .env file
load_dotenv()
# Ensure your .env file includes:
# OLLAMA_CHAT_MODEL=mistral
# OLLAMA_EMBED_MODEL=all-minilm


# Use PersistentClient to store data on disk.
chroma_client = chromadb.PersistentClient(path=r"C:\Users\frank\chroma_db")
try:
    collection = chroma_client.get_collection("site_pages")
except Exception:
    collection = chroma_client.create_collection("site_pages")

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Ollama's chat endpoint."""
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "mistral")
    system_prompt = (
        "You are an AI that extracts titles and summaries from documentation chunks.\n"
        "Return a JSON object with 'title' and 'summary' keys.\n"
        "For the title: If this seems like the start of a document, extract its title. "
        "If it's a middle chunk, derive a descriptive title.\n"
        "For the summary: Create a concise summary of the main points in this chunk.\n"
        "Keep both title and summary concise but informative."
    )
    try:
        response = ollama.chat(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ]
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from Ollama's embedding endpoint."""
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "all-minilm")
    try:
        response = ollama.embeddings(model=embed_model, prompt=text)
        return response["embedding"]
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return a zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted.get('title', ''),
        summary=extracted.get('summary', ''),
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into ChromaDB."""
    # Create a unique ID using the URL and chunk number.
    chunk_id = f"{chunk.url}-{chunk.chunk_number}"
    # Merge additional metadata
    metadata = {
        "url": chunk.url,
        "chunk_number": chunk.chunk_number,
        "title": chunk.title,
        "summary": chunk.summary,
        **chunk.metadata
    }
    try:
        # Wrap the blocking call with asyncio.to_thread
        await asyncio.to_thread(
            collection.add,
            ids=[chunk_id],
            embeddings=[chunk.embedding],
            documents=[chunk.content],
            metadatas=[metadata]
        )
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
    except Exception as e:
        print(f"Error inserting chunk into ChromaDB: {e}")

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_url(url: str):
        async with semaphore:
            try:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
            except Exception as e:
                if "EPIPE" in str(e):
                    print(f"EPIPE error while crawling {url}: {e}. Skipping URL.")
                    return
                else:
                    print(f"Error crawling {url}: {e}")
                    return

            if result.success:
                print(f"Successfully crawled: {url}")
                await process_and_store_document(url, result.markdown_v2.raw_markdown)
            else:
                print(f"Failed: {url} - Error: {result.error_message}")

    try:
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        try:
            await crawler.close()
        except Exception as e:
            if "EPIPE" in str(e):
                print("EPIPE error occurred during crawler shutdown, ignoring.")
            else:
                print("Error closing crawler:", e)

def get_fastapi_docs_urls() -> List[str]:
    """Get URLs from FastAPI docs sitemap."""
    sitemap_url = "https://fastapi.tiangolo.com/sitemap.xml"
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()
        
        root = ElementTree.fromstring(response.content)
        namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        return urls
    except Exception as e:
        print(f"Error fetching sitemap: {e}")
        return []

async def main():
    urls = get_fastapi_docs_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())
