from __future__ import annotations as _annotations

from dataclasses import dataclass
import asyncio
import httpx
import os
import sys
import json
from typing import Dict, Any, List, Optional

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from supabase import Client

from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer

def get_env_var(key: str, default: Optional[str] = None) -> str:
    return os.getenv(key, default)

load_dotenv()

llm = get_env_var('PRIMARY_MODEL')
base_url = get_env_var('BASE_URL')
is_ollama = "localhost" in base_url.lower()

model = OpenAIModel(llm, base_url=base_url)

EMBEDDING_MODEL_NAME = get_env_var("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

@dataclass
class FramerMotionDeps:
    supabase: Client
    reasoner_output: str

system_prompt = """
[ROLE AND CONTEXT]
You are a specialized AI agent engineer focused on answering Framer Motion applications.
You have comprehensive access to the Framer Motion documentation, including API references, usage guides, and implementation examples.
"""

# Create the agent using the Ollama-backed model.
framer_motion_coder = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=FramerMotionDeps,
    retries=2
)

@framer_motion_coder.system_prompt  
def add_reasoner_output(ctx: RunContext[str]) -> str:
    return f"""
Additional thoughts/instructions from the reasoner LLM:
{ctx.deps.reasoner_output}
"""

def compute_local_embedding(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings if isinstance(embeddings, list) else [embeddings]

@framer_motion_coder.tool
async def retrieve_relevant_documentation(ctx: RunContext[FramerMotionDeps], user_query: str) -> str:
    try:
        query_embedding = compute_local_embedding(user_query)
        
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 4,
                'filter': {'source': 'framer_motion_docs'}
            }
        ).execute()
        
        if not result.data:
            return "No relevant Framer Motion documentation found."
            
        formatted_chunks = []
        for doc in result.data:
            chunk_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunk_text)
            
        return "\n\n---\n\n".join(formatted_chunks)
        
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"Error retrieving documentation: {str(e)}"

async def list_documentation_pages_helper(supabase: Client) -> List[str]:
    try:
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'framer_motion_docs') \
            .execute()
        
        if not result.data:
            return []
        return sorted(set(doc['url'] for doc in result.data))
        
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@framer_motion_coder.tool
async def list_documentation_pages(ctx: RunContext[FramerMotionDeps]) -> List[str]:
    return await list_documentation_pages_helper(ctx.deps.supabase)

@framer_motion_coder.tool
async def get_page_content(ctx: RunContext[FramerMotionDeps], url: str) -> str:
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'framer_motion_docs') \
            .order('chunk_number') \
            .execute()
        
        if not result.data:
            return f"No content found for URL: {url}"
            
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        
        for chunk in result.data:
            formatted_content.append(chunk['content'])
            
        return "\n\n".join(formatted_content)[:20000]
        
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"
