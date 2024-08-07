"""
title: Beacon Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, langgraph, httpx, langchain, langchain_openai, pyowm, langchain-community, langchain-experimental, langchain-ollama, langchain-nomic, langchain-core==0.2.4, langgraph==0.0.64, langchain-community==0.2.3, langchain-openai==0.1.8, beautifulsoup4==4.12.3, termcolor==2.4.0, colorlog==6.8.2, fake-useragent==1.5.1, playwright==1.45.0, pypdf==4.2.0, llmsherpa==0.1.4, fastembed==0.3.4, faiss-cpu==1.8.0.post1, FlashRank==0.2.6

"""

# TO DO: Add tools to system prompt so LLM has awareness / usage rules on tools (see Ollama Functions docs), Add Langsmith evals and tracking, etc.
# Consider finetuning the sytnthesiztion step using a Single string, list of chat messages or a PromptValue all in one go
# Langsmith: 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
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
from langchain.memory import ChatMessageHistory
from jared import start_chat_session

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
                "LLAMAINDEX_MODEL_NAME": os.getenv("LLAMAINDEX_MODEL_NAME", "llama3.1"), #llama3-groq-tool-use
                "LLAMAINDEX_EMBEDDING_MODEL_NAME": os.getenv("LLAMAINDEX_EMBEDDING_MODEL_NAME", "nomic-embed-text"),
                "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "default-key"),
                "OPENWEATHERMAP_API_KEY": os.getenv("OPENWEATHERMAP_API_KEY", "default-key"),
            }
        )
        # Set LLM model to OpenAI
        #self.openai_model = ChatOpenAI(api_key=self.valves.OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
#     model="gpt-4o-mini",


    #     self.llm = OllamaFunctions(
    #                 model=self.valves.LLAMAINDEX_MODEL_NAME,
    #                 base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
    #                 format="json",  # Ensure JSON format is used for tool integration
    #                 temperature=0
    #             )

    #     #self.weather = OpenWeatherMapAPIWrapper()

    #     # self.llm_notools = ChatOllama(
    #     #             model="llama3",
    #     #             base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
    #     #             temperature=0,
    #     #             # other params...
    #     #         )
        
    #     self.llm_notools = OllamaFunctions(
    #                 model=self.valves.LLAMAINDEX_MODEL_NAME,
    #                 base_url=self.valves.LLAMAINDEX_OLLAMA_BASE_URL,
    #                 temperature=0
    #     )

    #     self.tools = [OpenWeatherMapQueryRun()]
    #    # self.functions = [convert_to_ollama_tool(t) for t in self.tools]
    #     self.llm = self.llm.bind_tools(self.tools)

    #     self.tool_executor = ToolExecutor(self.tools)


    #     # Define a LangChain graph
    #     # self.workflow = Graph()

    #     self.workflow = StateGraph(AgentState)


    #     self.workflow.add_node("agent", self.function_1)
    #     self.workflow.add_node("tool", self.function_2)

    #     # The conditional edge requires the following info below.
    #     # First, we define the start node. We use `agent`.
    #     # This means these are the edges taken after the `agent` node is called.
    #     # Next, we pass in the function that will determine which node is called next, in our case where_to_go().

    #     self.workflow.add_conditional_edges("agent", self.where_to_go,{   # Based on the return from where_to_go
    #                                                             # If return is "continue" then we call the tool node.
    #                                                             "continue": "tool",
    #                                                             # Otherwise we finish. END is a special node marking that the graph should finish.
    #                                                             "end": END
    #                                                         }
    #     )

    #     # We now add a normal edge from `tools` to `agent`.
    #     # This means that if `tool` is called, then it has to call the 'agent' next. 
    #     self.workflow.add_edge('tool', 'agent')

    #     # Basically, agent node has the option to call a tool node based on a condition, 
    #     # whereas tool node must call the agent in all cases based on this setup.

    #     self.workflow.set_entry_point("agent")


    #     self.app = self.workflow.compile()

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

        # self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        # self.index = VectorStoreIndex.from_documents(self.documents)
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass


    # def function_1(self, state):
    #     messages = state['messages']
    #     # Assuming messages[-1] should contain the necessary information
    #     last_message = messages[-1]
        
    #     # Debugging: Log details about the last message
    #     print(f"Calling function 1. Last Message Type: {type(last_message)} Content: {last_message.content}")
    #     print(f"Current Messages: {messages}")

    #     response = self.llm.invoke(messages)
    #     print(f"Response from function 1: {response}")       
    #     # ai_message = AIMessage(content=str(response))

    #     # Return the new state replacing the old messages with the function message
    #     return {"messages": [response]}

    # def function_2(self, state):
    #     messages = state['messages']
    #     agent_response = messages[-1]
    #     weather_data = self.weather.run(agent_response)
    #     state['messages'].append(weather_data)
    #     return state
    
    def function_1(self, state):
        messages = state['messages']
        # print(f"Initial and current Messages at start and each time function1 is called: {messages}")

        last_message = messages[-1]
        systemPrompt = messages[0].content
        systemPrompt = 'You are a helpful AI Assistant named Beacon.'
        # print(f"SYSTEM PROMPT BABY: {systemPrompt}")


        # print(f"Calling function 1. Last Message Type: {type(last_message)} Content: {getattr(last_message, 'content', 'No Content')}")

        # Check if the last message is a ToolMessage and handle it
        if isinstance(last_message, ToolMessage):

            # Find the last HumanMessage from the message history
            last_human_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
            human_content = last_human_message.content if last_human_message else "No previous human message found."
            # print(f"Human content: {human_content}")

            # Process the tool message content
            tool_response = last_message.content
            print(f"Current Messages during 2nd call: {messages}")
            # messages.append(tool_response)  # Add tool response to the context
            # print(f"Current Messages after appending: {messages}")
            
            #     # Construct a prompt template that asks the LLM to synthesize the conversation
            # synthesis_prompt = ChatPromptTemplate.from_messages([
            #     ("system", "Synthesize the conversation based on the following details: {details}"),
            #     ("human", last_message.content)
            # ])

            # # Format the messages for the prompt
            # formatted_prompt = synthesis_prompt.format_messages(details=last_message.content)
            # new_messages = {"messages": [HumanMessage(content="Please sythnesize that answer.")]}

            new_messages = [
                SystemMessage(content=systemPrompt),
                # HumanMessage(content=human_content),
                AIMessage(content="Here is some real-time information related to your query that I was able to retreive using a tool: " + tool_response),
                HumanMessage(content="Given this information, please answer my query: " + human_content),
                ]
            print(f"NEW prompt being used: {new_messages}")
            # Invoke the LLM with the new formatted prompt
            response_content = self.llm_notools.invoke(new_messages)
            



            # response_content = self.llm_notools.invoke(messages)
            # test_prompt = 'Synthesize this information: {response_content}'
            # print(f"New prompt being used : {test_prompt}")

            # response_content = self.llm.invoke([SystemMessage(content=str(test_prompt))])
            # response_content = AIMessage(content=str(response_content))





        else:
            # If not, invoke the LLM or handle other message types
            response_content = self.llm.invoke(messages)

        print(f"Final response from function 1: {response_content}")
        return {"messages": [response_content]}
    
    def function_2(self, state):
        messages = state['messages']
        last_message = messages[-1]  # Retrieve the last message which should have the tool call details

        # Verify that the last message has tool calls and select the last one
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[-1]

            tool_call_id = tool_call['id']  # Ensure the 'id' field is accessible and correct
            # tool_call.pop('run_manager', None)  # Remove the run_manager if present

            # Print the tool call details for debugging
            print(f"Tool Call Details: {tool_call}")

            # Formatting the invocation input correctly as a string or another supported format
            # Example: Pass a string or construct a suitable object as required by your llm.invoke method
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            # prompt = f"Run tool {tool_name} with arguments {tool_args}"  # Adjust format as needed
            # print(f"Test prompt for tool: {prompt}")


            # We construct an ToolInvocation from the function_call and pass in the tool name and the expected str input for OpenWeatherMap tool
            action = ToolInvocation(
                tool=tool_name,
                tool_input=tool_args,
            )

            # We call the tool_executor and get back a response
            response = self.tool_executor.invoke(action)

            # Print the response from the tool execution for debugging
            print(f"Response from tool execution: {response}")

            # Constructing ToolMessage with the required 'tool_call_id' field
            # function_message = AIMessage(content=str(response))

            function_message = ToolMessage(content=str(response), tool_call_id=tool_call_id)
            #function_message = FunctionMessage(content=str(response), name=action.tool)

            # Return the new state adding function message to messages list
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
        # print(f"Calling where to GO w last message: {last_message}")
        # if "function_call" in last_message.additional_kwargs:
        #     return "continue"
                # Check if the last message is an AIMessage and has tool calls
        if last_message.tool_calls:
            print(f"CONTINUING")
            return "continue"
        else:
            print(f"END")
            return "end"

    def prepare_pipeline_input(self, messages):
        # Initialize the list that will hold the message objects
        input_messages = []

        # Loop through the list of message dictionaries
        for message in messages:
            if message['role'] == 'system':
                beaconPrompt = "You are Beacon, an AI assistant dedicated to supporting professionals in the fields of human rights and international humanitarian law. You answer questions and complete tasks related to human rights issues, legal precedents, report drafting, and policy analysis.\n\nGuidelines for Response:\n- Accuracy and Relevance: Ensure all information is accurate, up-to-date, and specifically relevant to human rights and international humanitarian law. Cite reputable sources as necessary.\n- Depth of Analysis: Provide detailed and thorough analysis, addressing both the broad context and specific details of the query. Include relevant legal principles and precedents.\n- Jurisdictional and Contextual Awareness: Tailor your responses to reflect the specific legal and cultural context of the user’s jurisdiction and the applicable international conventions.\n- Clarity and Professionalism: Use precise legal terminology appropriately, but ensure explanations are clear and accessible to educated professionals.\n- Proactive Engagement: Suggest additional resources, further lines of inquiry, or strategic considerations that could assist the user in their work.\n- Output: Craft responses that are comprehensive, insightful, and directly applicable to the user's needs, aiding them in their professional duties effectively and efficiently." + "Here is some additional user information and context: " + message['content']

                # Add a system message when it exists
                input_messages.append(SystemMessage(content=beaconPrompt))
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
    #     # This is where you can add your custom RAG pipeline.
    #     # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.

    #     # If you'd like to check for title generation, you can add the following check
    #     # if body.get("title", False):
    #     #     print("Title Generation Request")
    #     #     return "Fake Title for Now"

        print(messages)
        print(user_message)

    #     # query_engine = self.index.as_query_engine(streaming=True)
    #     # response = query_engine.query(user_message)

    #     # return response.response_gen
    
    # # Invoke the LangGraph compiled app
    #     # inputs = {"messages": [user_message]}
        
    #     # inputs = {"messages": [
    #     #     # SystemMessage(content=messages[0].content), 
    #     #     HumanMessage(content=user_message)]}

    #     inputs = {"messages": self.prepare_pipeline_input(messages)}

    #     output = self.app.invoke(inputs)
    #     # print(f"FINAL OUTPUT: {output}")

    #     # Assuming output always contains a 'messages' list with at least one message
    #     last_message = output['messages'][-1].content if output['messages'] else "No AI message response found from pipeline"

    #     return last_message




        # Prepare the input for the chatbot
        # formatted_user_input = self.prepare_pipeline_input(messages)
        
        # Start the chat session and get the response
        response = start_chat_session(user_message)
        print(f"FINAL RESPONSE FROM JARED: {response}")

        return response