from __future__ import annotations as _annotations

import os
import sys
import json
import asyncio
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Annotated, TypedDict

from dotenv import load_dotenv
load_dotenv()

def get_env_var(key: str, default: Optional[str] = None) -> str:
    return os.getenv(key, default)

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from supabase import Client

from rag import framer_motion_coder, FramerMotionDeps

base_url = 'http://localhost:11411/v1'
primary_llm_model_name = 'llama3.1'
reasoner_llm_model_name = 'llama3.1'

is_ollama = "localhost" in base_url.lower()

reasoner_llm_model = OpenAIModel(reasoner_llm_model_name, base_url=base_url, api_key=api_key)
primary_llm_model = OpenAIModel(primary_llm_model_name, base_url=base_url, api_key=api_key)

reasoner = Agent(
    reasoner_llm_model,
    system_prompt='You are an expert at coding Applications and providing code with FramerMotion and defining scope.'
)

router_agent = Agent(
    primary_llm_model,
    system_prompt='Your job is to route the user message either to the end of the conversation or to continue coding.'
)

end_conversation_agent = Agent(
    primary_llm_model,
    system_prompt='Your job is to end the conversation by giving final instructions for executing the agent.'
)

if get_env_var("SUPABASE_URL"):
    supabase: Client = Client(
        get_env_var("SUPABASE_URL"),
        get_env_var("SUPABASE_SERVICE_KEY")
    )
else:
    supabase = None

class AgentState(TypedDict):
    latest_user_message: str
    messages: Annotated[List[bytes], lambda x, y: x + y]
    scope: str

async def define_scope_with_reasoner(state: AgentState):
    try:
        result = supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'framer_motion_docs') \
            .execute()
        documentation_pages = sorted(set(doc['url'] for doc in result.data)) if result.data else []
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        documentation_pages = []
    
    documentation_pages_str = "\n".join(documentation_pages)

    prompt = f"""
User AI Agent Request: {state['latest_user_message']}

Provide proper applications or code.

The user specifically wants to use the Framer Motion docs.
Available docs:
{documentation_pages_str}
"""
    result = await reasoner.run(prompt)
    scope = result.data

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    scope_path = os.path.join(parent_dir, "workbench", "scope.md")
    os.makedirs(os.path.join(parent_dir, "workbench"), exist_ok=True)
    with open(scope_path, "w", encoding="utf-8") as f:
        f.write(scope)
    
    return {"scope": scope}

async def coder_agent(state: AgentState, writer):
    deps = FramerMotionDeps(
        supabase=supabase,
        reasoner_output=state['scope']
    )
    message_history: List[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    result = await framer_motion_coder.run(state['latest_user_message'], deps=deps, message_history=message_history)
    writer(result.data)

    return {"messages": [result.new_messages_json()]}

def get_next_user_message(state: AgentState):
    value = interrupt({})
    return {"latest_user_message": value}

async def route_user_message(state: AgentState):
    prompt = f"""
The user said: {state['latest_user_message']}

If the user wants to end the conversation, respond "finish_conversation".
Otherwise, respond "coder_agent".
"""
    result = await router_agent.run(prompt)
    next_action = result.data.strip()
    if next_action == "finish_conversation":
        return "finish_conversation"
    else:
        return "coder_agent"

async def finish_conversation(state: AgentState, writer):
    message_history: List[ModelMessage] = []
    for message_row in state['messages']:
        message_history.extend(ModelMessagesTypeAdapter.validate_json(message_row))

    result = await end_conversation_agent.run(state['latest_user_message'], message_history=message_history)
    writer(result.data)
    return {"messages": [result.new_messages_json()]}

builder = StateGraph(AgentState)
builder.add_node("define_scope_with_reasoner", define_scope_with_reasoner)
builder.add_node("coder_agent", coder_agent)
builder.add_node("get_next_user_message", get_next_user_message)
builder.add_node("finish_conversation", finish_conversation)

builder.add_edge(START, "define_scope_with_reasoner")
builder.add_edge("define_scope_with_reasoner", "coder_agent")
builder.add_edge("coder_agent", "get_next_user_message")
builder.add_conditional_edges(
    "get_next_user_message",
    route_user_message,
    {"coder_agent": "coder_agent", "finish_conversation": "finish_conversation"}
)
builder.add_edge("finish_conversation", END)

memory = MemorySaver()
agentic_flow = builder.compile(checkpointer=memory)
