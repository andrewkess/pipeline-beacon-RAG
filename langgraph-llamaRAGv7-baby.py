"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, langgraph, httpx, langchain, langchain_openai, pyowm, langchain-community, langchain-experimental, langchain-ollama
"""

from typing import List, Union, Generator, Iterator, TypedDict, Annotated, Sequence
from schemas import OpenAIChatMessage
import operator
import os
from langgraph.graph import Graph, StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, FunctionMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
import json
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_ollama import ChatOllama
import logging
from datetime import datetime
import time

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        OPENAI_API_KEY: str
        OPENWEATHERMAP_API_KEY: str

    def __init__(self):
        self.documents = None
        self.index = None

        # assign AgentState as an empty dict
        self.AgentState = {}

        # messages key will be assigned as an empty array. We will append new messages as we pass along nodes. 
        self.AgentState["messages"] = []

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text-v1.5"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "default-key"),
                "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", "default-key"),
            }
        )

        self.llm = OllamaFunctions(
                    model=self.valves.LLAMAINDEX_MODEL_NAME,
                    base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
                    format="json",
                    temperature=0
                )

        self.llm_notools = OllamaFunctions(
                    model=self.valves.LLAMAINDEX_MODEL_NAME,
                    base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
                    temperature=0
        )

        self.tools = [OpenWeatherMapQueryRun()]
        self.llm = self.llm.bind_tools(self.tools)
        self.tool_executor = ToolExecutor(self.tools)

        self.workflow = StateGraph(AgentState)

        self.workflow.add_node("agent", self.function_1)
        self.workflow.add_node("tool", self.function_2)

        self.workflow.add_conditional_edges("agent", self.where_to_go, {
                                                                "continue": "tool",
                                                                "end": END
                                                            }
        )

        self.workflow.add_edge('tool', 'agent')
        self.workflow.set_entry_point("agent")

        self.app = self.workflow.compile()

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

        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)

    async def on_shutdown(self):
        pass

    def function_1(self, state):
        messages = state['messages']
        last_message = messages[-1]
        systemPrompt = 'You are a helpful AI Assistant named Beacon.'

        if isinstance(last_message, ToolMessage):
            last_human_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            human_content = last_human_message.content if last_human_message else "No previous human message found."

            tool_response = last_message.content
            new_messages = [
                SystemMessage(content=systemPrompt),
                AIMessage(content="Here is some real-time information related to your query that I was able to retrieve using a tool: " + tool_response),
                HumanMessage(content="Given this information, please answer my query: " + human_content),
            ]
            response_content = self.llm_notools.invoke(new_messages)
        else:
            response_content = self.llm.invoke(messages)

        return {"messages": [response_content]}

    def function_2(self, state):
        messages = state['messages']
        last_message = messages[-1]

        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[-1]
            tool_call_id = tool_call['id']

            tool_name = tool_call['name']
            tool_args = tool_call['args']

            action = ToolInvocation(
                tool=tool_name,
                tool_input=tool_args,
            )

            response = self.tool_executor.invoke(action)

            function_message = ToolMessage(content=str(response), tool_call_id=tool_call_id)

            return {"messages": [function_message]}

    def where_to_go(self, state):
        messages = state['messages']
        last_message = messages[-1]

        if last_message.tool_calls:
            return "continue"
        else:
            return "end"

    def prepare_pipeline_input(self, messages):
        input_messages = []

        for message in messages:
            if message['role'] == 'system':
                beaconPrompt = "You are Beacon, an AI assistant dedicated to supporting professionals in the fields of human rights and international humanitarian law. You answer questions and complete tasks related to human rights issues, legal precedents, report drafting, and policy analysis.\n\nGuidelines for Response:\n- Accuracy and Relevance: Ensure all information is accurate, up-to-date, and specifically relevant to human rights and international humanitarian law. Cite reputable sources as necessary.\n- Depth of Analysis: Provide detailed and thorough analysis, addressing both the broad context and specific details of the query. Include relevant legal principles and precedents.\n- Jurisdictional and Contextual Awareness: Tailor your responses to reflect the specific legal and cultural context of the user’s jurisdiction and the applicable international conventions.\n- Clarity and Professionalism: Use precise legal terminology appropriately, but ensure explanations are clear and accessible to educated professionals.\n- Proactive Engagement: Suggest additional resources, further lines of inquiry, or strategic considerations that could assist the user in their work.\n- Output: Craft responses that are comprehensive, insightful, and directly applicable to the user's needs, aiding them in their professional duties effectively and efficiently." + "Here is some additional user information and context: " + message['content']

                input_messages.append(SystemMessage(content=beaconPrompt))
            elif message['role'] == 'user':
                input_messages.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                input_messages.append(AIMessage(content=message['content']))

        return input_messages

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        inputs = {"messages": self.prepare_pipeline_input(messages)}

        output = self.app.invoke(inputs)

        last_message = output['messages'][-1].content if output['messages'] else "No AI message response found from pipeline"

        return last_message
