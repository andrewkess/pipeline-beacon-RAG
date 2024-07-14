"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, langgraph, httpx, langchain, langchain_openai
"""

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os
from langgraph.graph import Graph
from langchain_openai import ChatOpenAI
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str
        OPENAI_API_KEY: str

    def __init__(self):
        self.documents = None
        self.index = None

        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "default-key"),
            }
        )
        # Set LLM model to OpenAI
        self.openai_model = ChatOpenAI(api_key=self.valves.OPENAI_API_KEY)

        # Define a LangChain graph
        self.workflow = Graph()

        self.workflow.add_node("node_1", self.function_1_using_openai)
        self.workflow.add_node("node_2", self.function_2)

        self.workflow.add_edge('node_1', 'node_2')

        self.workflow.set_entry_point("node_1")
        self.workflow.set_finish_point("node_2")

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

    def function_1(self, input_1):
        return input_1 + " Hi "

    def function_1_using_openai(self, input_1):
        response = self.openai_model.invoke(input_1)
        return response.content if response else "No response received"

    def function_2(self, input_2):
        return input_2 + "there"

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
