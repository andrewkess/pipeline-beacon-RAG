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
from langgraph.graph import Graph, StateGraph, END
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, FunctionMessage, AIMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_community.tools.openweathermap import OpenWeatherMapQueryRun
import json
from langgraph.prebuilt import ToolExecutor, ToolInvocation

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
        self.openai_model = ChatOpenAI(api_key=self.valves.OPENAI_API_KEY, model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
#     model="gpt-4o-mini",

        #self.weather = OpenWeatherMapAPIWrapper()

        self.tools = [OpenWeatherMapQueryRun()]
        self.functions = [convert_to_openai_function(t) for t in self.tools]
        self.openai_model = self.openai_model.bind_functions(self.functions)

        self.tool_executor = ToolExecutor(self.tools)


        # Define a LangChain graph
        # self.workflow = Graph()

        self.workflow = StateGraph(AgentState)


        self.workflow.add_node("agent", self.function_1)
        self.workflow.add_node("tool", self.function_2)

        # The conditional edge requires the following info below.
        # First, we define the start node. We use `agent`.
        # This means these are the edges taken after the `agent` node is called.
        # Next, we pass in the function that will determine which node is called next, in our case where_to_go().

        self.workflow.add_conditional_edges("agent", self.where_to_go,{   # Based on the return from where_to_go
                                                                # If return is "continue" then we call the tool node.
                                                                "continue": "tool",
                                                                # Otherwise we finish. END is a special node marking that the graph should finish.
                                                                "end": END
                                                            }
        )

        # We now add a normal edge from `tools` to `agent`.
        # This means that if `tool` is called, then it has to call the 'agent' next. 
        self.workflow.add_edge('tool', 'agent')

        # Basically, agent node has the option to call a tool node based on a condition, 
        # whereas tool node must call the agent in all cases based on this setup.

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

        # This function is called when the server is started.
        global documents, index

        self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    # def function_1_using_openai(self, state):
    #     messages = state['messages']
    #     user_input = messages[-1]
    #     complete_query = "Your task is to provide only the city name based on the user query. \
    #                     Nothing more, just the city name mentioned. Following is the user query: " + user_input
    #     response = self.openai_model.invoke(complete_query)
    #     state['messages'].append(response.content) # appending AIMessage response to the AgentState
    #     return state
    
    def function_1(self, state):
        messages = state['messages']
        response = self.openai_model.invoke(messages)
        return {"messages": [response]}    

    # def function_2(self, state):
    #     messages = state['messages']
    #     agent_response = messages[-1]
    #     weather_data = self.weather.run(agent_response)
    #     state['messages'].append(weather_data)
    #     return state

    def function_2(self, state):
        messages = state['messages']
        last_message = messages[-1] # this has the query we need to send to the tool provided by the agent

        parsed_tool_input = json.loads(last_message.additional_kwargs["function_call"]["arguments"])

        # We construct an ToolInvocation from the function_call and pass in the tool name and the expected str input for OpenWeatherMap tool
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=parsed_tool_input['__arg1'],
        )
        
        # We call the tool_executor and get back a response
        response = self.tool_executor.invoke(action)

        # We use the response to create a FunctionMessage
        function_message = FunctionMessage(content=str(response), name=action.tool)

        # We return a list, because this will get added to the existing list
        return {"messages": [function_message]}

    # def function_3(self, state):
    #     messages = state['messages']
    #     user_input = messages[0]
    #     available_info = messages[-1]
    #     agent2_query = "Your task is to provide info concisely based on the user query and the available information from the internet. \
    #                         Following is the user query: " + user_input + " Available information: " + available_info
    #     response = self.openai_model.invoke(agent2_query)
    #     return response.content

    def where_to_go(self, state):
        messages = state['messages']
        last_message = messages[-1]
        
        if "function_call" in last_message.additional_kwargs:
            return "continue"
        else:
            return "end"

    def prepare_pipeline_input(self, messages):
        # Initialize the list that will hold the message objects
        input_messages = []

        # Loop through the list of message dictionaries
        for message in messages:
            if message['role'] == 'system':
                # Add a system message when it exists
                input_messages.append(SystemMessage(content=message['content']))
            elif message['role'] == 'user':
                # Always add the user message
                input_messages.append(HumanMessage(content=message['content']))
            elif message['role'] == 'assistant':
                # Always add the user message
                input_messages.append(AIMessage(content=message['content']))

        return input_messages

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

        # If you'd like to check for title generation, you can add the following check
        # if body.get("title", False):
        #     print("Title Generation Request")
        #     return "Fake Title for Now"

        print(messages)
        print(user_message)

        # query_engine = self.index.as_query_engine(streaming=True)
        # response = query_engine.query(user_message)

        # return response.response_gen
    
    # Invoke the LangGraph compiled app
        # inputs = {"messages": [user_message]}
        
        # inputs = {"messages": [
        #     # SystemMessage(content=messages[0].content), 
        #     HumanMessage(content=user_message)]}

        inputs = {"messages": self.prepare_pipeline_input(messages)}

        output = self.app.invoke(inputs)
        print(output)

        # Assuming output always contains a 'messages' list with at least one message
        last_message = output['messages'][-1].content if output['messages'] else "No AI message response found from pipeline"

        return last_message

        # # Initialize a variable to hold the last AI response
        # last_ai_response = None

        # # Initialize a variable to hold the AI response
        # ai_response = None

        # # Check if 'messages' key exists and iterate over messages
        # if 'messages' in output:
        #     for message in output['messages']:
        #         # Check if the message is an AIMessage
        #         if isinstance(message, AIMessage):
        #             ai_response = message.content
        #             # break  # Assuming you only need the first AIMessage content

        # # Return the AI response or a default message if no AIMessage was found
        # return ai_response if ai_response else "No AI response found"


        # # Check if 'messages' key exists and iterate over messages
        # if 'messages' in output:
        #     for message in output['messages']:
        #         # Assuming 'type' in message dict helps identify AIMessage
        #         if message.get('type') == 'AIMessage':
        #             last_ai_response = message['content']

        # # Return the last AI response or a default message if no AIMessage was found
        # return last_ai_response if last_ai_response else "No AI response found"
