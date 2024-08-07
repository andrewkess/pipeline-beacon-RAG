"""
title: Jar3d Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for interacting with the Jar3d agent using the Langgraph framework.
requirements: langgraph, termcolor, pydantic, datetime
"""

from typing import List, Union, Generator, Iterator, Any, Dict
from schemas import OpenAIChatMessage
import os
import json
import re
import textwrap
import logging
from termcolor import colored
from datetime import datetime
from typing import TypedDict, Annotated
from pydantic import BaseModel

from langgraph.graph.message import add_messages
from base_agent import BaseAgent
from utils.read_markdown import read_markdown_file
from utils.logging import log_function, setup_logging
from langgraph.graph import StateGraph

setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class MessageDict(TypedDict):
    role: str
    content: str

class State(TypedDict):
    meta_prompt: Annotated[List[MessageDict], add_messages]
    conversation_history: Annotated[List[dict], add_messages]
    requirements_gathering: Annotated[List[str], add_messages]
    expert_plan: str
    expert_research: Annotated[List[str], add_messages]
    expert_writing: str
    user_input: Annotated[List[str], add_messages]
    previous_search_queries: Annotated[List[dict], add_messages]
    router_decision: bool
    chat_limit: int
    chat_finished: bool
    recursion_limit: int

def chat_counter(state: State) -> State:
    chat_limit = state.get("chat_limit")
    if chat_limit is None:
        chat_limit = 0
    chat_limit += 1
    state["chat_limit"] = chat_limit
    return chat_limit

def routing_function(state: State) -> str:
    if state["router_decision"]:
        return "no_tool_expert"
    else:
        return "tool_expert"

def set_chat_finished(state: State) -> bool:
    state["chat_finished"] = True
    final_response = state["meta_prompt"][-1].content
    final_response_formatted = re.sub(r'^```python[\s\S]*?```\s*', '', final_response, flags=re.MULTILINE)
    final_response_formatted = final_response_formatted.lstrip()
    print(colored(f"\n\n Jar3d👩‍💻: {final_response_formatted}", 'cyan'))

    return state

class Jar3d(BaseAgent[State]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        super().__init__(model, server, temperature, model_endpoint, stop)
        self.llm = self.get_llm(json_model=False)

    def get_prompt(self, state:State = None) -> str:
        system_prompt = read_markdown_file('prompt_engineering/jar3d_requirements_prompt.md')
        return system_prompt
        
    def process_response(self, response: Any, user_input: str, state: State = None) -> Dict[str, List[MessageDict]]:
        user_input = None
        updates_conversation_history = {
            "requirements_gathering": [
                {"role": "user", "content": f"{user_input}"},
                {"role": "assistant", "content": str(response)}
            ]
        }
        return updates_conversation_history
    
    def get_conv_history(self, state: State) -> str:
        conversation_history = state.get('requirements_gathering', [])
        return conversation_history
    
    def get_user_input(self) -> str:
        user_input = input("Enter your query: ")
        return user_input
    
    def get_guided_json(self, state: State) -> Dict[str, Any]:
        pass

    def use_tool(self) -> Any:
        pass

    @log_function(logger)
    def run(self, state: State) -> State:
        history = self.get_conv_history(state)
        user_input = state.get("user_input")

        system_prompt = self.get_prompt()
        user_input = f"previous conversation: {history}\n {system_prompt}\n cogor {user_input}"

        while True:
            history = self.get_conv_history(state)
            state = self.invoke(state=state, user_input=user_input)
            response = state['requirements_gathering'][-1]["content"]
            response = re.sub(r'^```python[\s\S]*?```\s*', '', response, flags=re.MULTILINE)
            response = response.lstrip()

            print("\n" + "="*80)  # Print a separator line
            print(colored("Jar3d:", 'cyan', attrs=['bold']))
            
            # Wrap the text to a specified width (e.g., 70 characters)
            wrapped_text = textwrap.fill(response, width=70)
            
            # Print each line with proper indentation
            for line in wrapped_text.split('\n'):
                print(colored("  " + line, 'green'))
            
            print("="*80 + "\n")  #
            user_input = self.get_user_input()
            
            if user_input == "/end":
                break
            
            user_input = f"cogor {user_input}"

        state = self.invoke(state=state, user_input=user_input)
        response = state['requirements_gathering'][-1]["content"]
        response = re.sub(r'^```python[\s\S]*?```\s*', '', response, flags=re.MULTILINE)
        response = response.lstrip()

        print("\n" + "="*80)  # Print a separator line
        print(colored("Jar3d:", 'cyan', attrs=['bold']))
        for line in wrapped_text.split('\n'):
                print(colored("  " + line, 'green'))
            
        print("="*80 + "\n")

        return state


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    async def on_startup(self):
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        Settings.embed_model = OllamaEmbedding(
            model_name=self.valves.LLAMAINDEX_EMBEDDING_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )
        Settings.llm = Ollama(
            model=self.valves.LLAMAINDEX_MODEL_NAME,
            base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
        )

        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # Initialize the state
        state = State(
            meta_prompt=[],
            conversation_history=[],
            requirements_gathering=[],
            expert_plan="",
            expert_research=[],
            expert_writing="",
            user_input=[user_message],
            previous_search_queries=[],
            router_decision=False,
            chat_limit=0,
            chat_finished=False,
            recursion_limit=5,
        )

        def start_chat_session(user_input: str, state: State):
            agent_kwargs = {
                "model": "claude-3-5-sonnet-20240620",
                "server": "claude",
                "temperature": 0.2
            }

            graph = StateGraph(State)

            graph.add_node("jar3d", lambda state: Jar3d(**agent_kwargs).run(state=state))
            graph.add_node("end_chat", lambda state: set_chat_finished(state))

            graph.set_entry_point("jar3d")
            graph.set_finish_point("end_chat")

            workflow = graph.compile()

            limit = {"recursion_limit": state["recursion_limit"] + 10}  # Required as a buffer.

            for event in workflow.stream(state, limit):
                yield event

        # Stream the response
        return start_chat_session(user_message, state)

