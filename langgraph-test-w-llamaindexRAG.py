"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, langgraph, httpx, langchain, langchain_openai, pyowm, langchain-community
"""

from typing import List, Union, Generator, Iterator, TypedDict, Annotated, Sequence
from schemas import OpenAIChatMessage
import operator
import os
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.messages import BaseMessage

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
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "default-key"),
                "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", "default-key"),
            }
        )
        # Set LLM model to OpenAI
        self.openai_model = ChatOpenAI(api_key=self.valves.OPENAI_API_KEY)

        self.weather = OpenWeatherMapAPIWrapper()

        # Define a LangChain graph
        self.workflow = Graph()

        self.workflow.add_node("agent", self.function_1_using_openai)
        self.workflow.add_node("tool", self.function_2)
        self.workflow.add_node("responder", self.function_3)

        self.workflow.add_edge('agent', 'tool')
        self.workflow.add_edge('tool', 'responder')

        self.workflow.set_entry_point("agent")
        self.workflow.set_finish_point("responder")

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

        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    def function_1_using_openai(self, state):
        messages = state['messages']
        user_input = messages[-1]
        complete_query = "Your task is to provide only the city name based on the user query. \
                        Nothing more, just the city name mentioned. Following is the user query: " + user_input
        response = self.openai_model.invoke(complete_query)
        state['messages'].append(response.content) # appending AIMessage response to the AgentState
        return state

    def function_2(self, state):
        messages = state['messages']
        agent_response = messages[-1]
        weather_data = self.weather.run(agent_response)
        state['messages'].append(weather_data)
        return state

    def function_3(self, state):
        messages = state['messages']
        user_input = messages[0]
        available_info = messages[-1]
        agent2_query = "Your task is to provide info concisely based on the user query and the available information from the internet. \
                            Following is the user query: " + user_input + " Available information: " + available_info
        response = self.openai_model.invoke(agent2_query)
        return response.content

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)

        # return response.response_gen
    
    # Invoke the LangGraph compiled app
        output = self.app.invoke(user_message)
        return output
