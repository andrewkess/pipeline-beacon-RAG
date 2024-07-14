"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: google-search-results, langchain, langchain_core, langchain_openai, langchain_community, llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""

import re

from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import LLMChain
from langchain.prompts import StringPromptTemplate
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


import os
#os.environ["SERPAPI_API_KEY"] = os.getenv("SERPAPI_API_KEY")
#os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# Get the environment variables
#serpapi_key = os.getenv("SERPAPI_API_KEY")
#openai_key = os.getenv("OPENAI_API_KEY")




from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from pydantic import BaseModel


class Pipeline:

    class Valves(BaseModel):
        LLAMAINDEX_OLLAMA_BASE_URL: str
        LLAMAINDEX_MODEL_NAME: str
        LLAMAINDEX_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.documents = None
        self.index = None
        self.tools = self.setup_tools()  # Setting up tools during pipeline initialization
        self.search_api = SerpAPIWrapper()  # Initialize the SerpAPIWrapper here
        self.valves = self.Valves(
            **{
                "LLAMAINDEX_OLLAMA_BASE_URL": os.getenv("LLAMAINDEX_OLLAMA_BASE_URL", "http://localhost:11434"),
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3"),
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            }
        )

    def setup_tools(self):
        # This method will setup all your tools, including fake ones for demonstration
        search_tool = Tool(
            name="Search",
            func=self.search_function,  # Define this function to perform actual searches
            description="Useful for answering questions about current events"
        )

        fake_tools = [
            Tool(
                name=f"FakeTool-{i}",
                func=self.fake_function,  # Define this as a placeholder function
                description=f"Placeholder functionality {i}"
            ) for i in range(99)  # Example range, adjust as needed
        ]

        return [search_tool] + fake_tools

    def search_function(self, query: str) -> str:
        # Implement the actual search logic here
        # Implement the actual search logic here using SerpAPIWrapper
        search_results = self.search_api.run(query)
        return search_results

    def fake_function(self, query: str) -> str:
        # Placeholder function that does nothing useful
        return "This is a fake response"

    def setup_embeddings(self):
        # Initialize OpenAI embeddings model
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

        # Create Document objects from tool descriptions
        documents = [Document(page_content=tool.description) for tool in self.tools]

        # Create embeddings for each document
        embeddings = [self.embeddings_model.embed(document.page_content) for document in documents]

        # Initialize FAISS vector store with these embeddings
        self.vector_store = FAISS()
        for embed, tool in zip(embeddings, self.tools):
            self.vector_store.add(embed, tool)

    def retrieve_tools(self, query: str):
        # Embed the query
        query_embedding = self.embeddings_model.embed(query)

        # Retrieve tools based on the query embedding
        similar_tools, _ = self.vector_store.search(query_embedding, k=5)  # adjust k based on how many tools you want to retrieve

        return similar_tools

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

        self.setup_embeddings()

        # This function is called when the server is started.
        global documents, index
        
        #unused for now, as i dont want to read data from a RAG pipeline just yet. For now, i just want to make an agent with tooling
        #self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        #self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        print(messages)
        print(user_message)


        #unused for now, as i dont want to  use a RAG pipeline just yet. For now, i just want to make an agent with tooling

        #query_engine = self.index.as_query_engine(streaming=True)
        #response = query_engine.query(user_message)

        #return response.response_gen

        # Retrieve tools relevant to the user message
        relevant_tools = self.retrieve_tools(user_message)
        # Use these tools to handle the user's query
        responses = [tool.func(user_message) for tool in relevant_tools]
        return " ".join(responses)  # Adjust based on how you want to combine responses