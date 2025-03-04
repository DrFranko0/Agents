import os
import sys
import asyncio
import threading
import subprocess
import requests
import json
from typing import List, Dict, Any, Optional, Callable
from xml.etree import ElementTree
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
import re
import html2text

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_env_var

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

# Retain base_url for other services if needed; the openai API is no longer used.
base_url = get_env_var('BASE_URL') or 'https://api.openai.com/v1'
# Use an open-source embedding model from SentenceTransformers
embedding_model = 'all-MiniLM-L6-v2'

supabase: Client = create_client(
    get_env_var("SUPABASE_URL"),
    get_env_var("SUPABASE_SERVICE_KEY")
)

html_converter = html2text.HTML2Text()
html_converter.ignore_links = False
html_converter.ignore_images = False
html_converter.ignore_tables = False
html_converter.body_width = 0

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

class CrawlProgressTracker:    
    def __init__(self, 
                 progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.progress_callback = progress_callback
        self.urls_found = 0
        self.urls_processed = 0
        self.urls_succeeded = 0
        self.urls_failed = 0
        self.chunks_stored = 0
        self.logs = []
        self.is_running = False
        self.start_time = None
        self.end_time = None
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        print(message)
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def start(self):
        self.is_running = True
        self.start_time = datetime.now()
        self.log("Crawling process started")
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def complete(self):
        self.is_running = False
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time if self.start_time else None
        duration_str = str(duration).split('.')[0] if duration else "unknown"
        self.log(f"Crawling process completed in {duration_str}")
        if self.progress_callback:
            self.progress_callback(self.get_status())
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "is_running": self.is_running,
            "urls_found": self.urls_found,
            "urls_processed": self.urls_processed,
            "urls_succeeded": self.urls_succeeded,
            "urls_failed": self.urls_failed,
            "chunks_stored": self.chunks_stored,
            "progress_percentage": (self.urls_processed / self.urls_found * 100) if self.urls_found > 0 else 0,
            "logs": self.logs,
            "start_time": self.start_time,
            "end_time": self.end_time
        }
    
    @property
    def is_completed(self) -> bool:
        return not self.is_running and self.end_time is not None
    
    @property
    def is_successful(self) -> bool:
        return self.is_completed and self.urls_failed == 0 and self.urls_succeeded > 0

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
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

def sync_get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    title_match = re.search(r"^(?:#)+\s*(.+)", chunk, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        title = chunk.split(".")[0].strip() if "." in chunk else chunk[:50].strip()
    
    try:
        from transformers import pipeline
        if not hasattr(sync_get_title_and_summary, "summarizer"):
            sync_get_title_and_summary.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summarizer = sync_get_title_and_summary.summarizer
        
        text_for_summary = chunk if len(chunk) < 1024 else chunk[:1024]
        summary_output = summarizer(text_for_summary, max_length=130, min_length=30, do_sample=False)
        summary = summary_output[0]['summary_text']
    except Exception as e:
        print(f"Error in summarization: {e}")
        summary = "Summary not available"
    
    return {"title": title, "summary": summary}

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    return await asyncio.to_thread(sync_get_title_and_summary, chunk, url)

def sync_get_embedding(text: str) -> List[float]:
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(sync_get_embedding, "model"):
            sync_get_embedding.model = SentenceTransformer(embedding_model)
        model = sync_get_embedding.model
        embedding = model.encode(text).tolist()
        return embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero vector (384 dimensions for the MiniLM model) if embedding fails
        return [0.0] * 384

async def get_embedding(text: str) -> List[float]:
    return await asyncio.to_thread(sync_get_embedding, text)

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    metadata = {
        "source": "framer_motion_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str, tracker: Optional[CrawlProgressTracker] = None):
    chunks = chunk_text(markdown)
    
    if tracker:
        tracker.log(f"Split document into {len(chunks)} chunks for {url}")
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Split document into {len(chunks)} chunks for {url}")
    
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    
    if tracker:
        tracker.log(f"Processed {len(processed_chunks)} chunks for {url}")
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Processed {len(processed_chunks)} chunks for {url}")
    
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)
    
    if tracker:
        tracker.chunks_stored += len(processed_chunks)
        tracker.log(f"Stored {len(processed_chunks)} chunks for {url}")
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Stored {len(processed_chunks)} chunks for {url}")

def fetch_url_content(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        markdown = html_converter.handle(response.text)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        return markdown
    except Exception as e:
        raise Exception(f"Error fetching {url}: {str(e)}")

async def crawl_parallel_with_requests(urls: List[str], tracker: Optional[CrawlProgressTracker] = None, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_url(url: str):
        async with semaphore:
            if tracker:
                tracker.log(f"Crawling: {url}")
                if tracker.progress_callback:
                    tracker.progress_callback(tracker.get_status())
            else:
                print(f"Crawling: {url}")
            try:
                loop = asyncio.get_running_loop()
                if tracker:
                    tracker.log(f"Fetching content from: {url}")
                else:
                    print(f"Fetching content from: {url}")
                markdown = await loop.run_in_executor(None, fetch_url_content, url)
                
                if markdown:
                    if tracker:
                        tracker.urls_succeeded += 1
                        tracker.log(f"Successfully crawled: {url}")
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, markdown, tracker)
                else:
                    if tracker:
                        tracker.urls_failed += 1
                        tracker.log(f"Failed: {url} - No content retrieved")
                        if tracker.progress_callback:
                            tracker.progress_callback(tracker.get_status())
                    else:
                        print(f"Failed: {url} - No content retrieved")
            except Exception as e:
                if tracker:
                    tracker.urls_failed += 1
                    tracker.log(f"Error processing {url}: {str(e)}")
                    if tracker.progress_callback:
                        tracker.progress_callback(tracker.get_status())
                else:
                    print(f"Error processing {url}: {str(e)}")
            finally:
                if tracker:
                    tracker.urls_processed += 1
                    if tracker.progress_callback:
                        tracker.progress_callback(tracker.get_status())
    
    if tracker:
        tracker.log(f"Processing {len(urls)} URLs with concurrency {max_concurrent}")
        if tracker.progress_callback:
            tracker.progress_callback(tracker.get_status())
    else:
        print(f"Processing {len(urls)} URLs with concurrency {max_concurrent}")
    await asyncio.gather(*[process_url(url) for url in urls])

def get_framer_motion_docs_urls() -> List[str]:
    sitemap_url = "https://motion.dev/sitemap.xml"
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

async def clear_existing_records():
    try:
        result = supabase.table("site_pages").delete().eq("metadata->>source", "framer_motion_docs").execute()
        print("Cleared existing framer_motion_docs records from site_pages")
        return result
    except Exception as e:
        print(f"Error clearing existing records: {e}")
        return None

async def main_with_requests(tracker: Optional[CrawlProgressTracker] = None):
    try:
        if tracker:
            tracker.start()
        else:
            print("Starting crawling process...")
        
        if tracker:
            tracker.log("Clearing existing Framer Motion docs records...")
        else:
            print("Clearing existing Framer Motion docs records...")
        await clear_existing_records()
        
        if tracker:
            tracker.log("Existing records cleared")
        else:
            print("Existing records cleared")
        
        if tracker:
            tracker.log("Fetching URLs from Framer Motion sitemap...")
        else:
            print("Fetching URLs from Framer Motion sitemap...")
        urls = get_framer_motion_docs_urls()
        
        if not urls:
            if tracker:
                tracker.log("No URLs found to crawl")
                tracker.complete()
            else:
                print("No URLs found to crawl")
            return
        
        if tracker:
            tracker.urls_found = len(urls)
            tracker.log(f"Found {len(urls)} URLs to crawl")
        else:
            print(f"Found {len(urls)} URLs to crawl")
        
        await crawl_parallel_with_requests(urls, tracker)
        
        if tracker:
            tracker.complete()
        else:
            print("Crawling process completed")
            
    except Exception as e:
        if tracker:
            tracker.log(f"Error in crawling process: {str(e)}")
            tracker.complete()
        else:
            print(f"Error in crawling process: {str(e)}")

def start_crawl_with_requests(progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> CrawlProgressTracker:
    tracker = CrawlProgressTracker(progress_callback)
    
    def run_crawl():
        try:
            asyncio.run(main_with_requests(tracker))
        except Exception as e:
            print(f"Error in crawl thread: {e}")
            tracker.log(f"Thread error: {str(e)}")
            tracker.complete()
    thread = threading.Thread(target=run_crawl)
    thread.daemon = True
    thread.start()
    return tracker

if __name__ == "__main__":    
    print("Starting crawler...")
    asyncio.run(main_with_requests())
    print("Crawler finished.")
