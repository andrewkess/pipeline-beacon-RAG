"""
title: Jared Pipeline
author: 
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama, langgraph, httpx, langchain, langchain_openai, pyowm, langchain-community, langchain-experimental, langchain-ollama, langchain-nomic, langchain-core==0.2.4, langgraph==0.0.64, langchain-community==0.2.3, langchain-openai==0.1.8, beautifulsoup4==4.12.3, termcolor==2.4.0, colorlog==6.8.2, fake-useragent==1.5.1, playwright==1.45.0, pypdf==4.2.0, llmsherpa==0.1.4, fastembed==0.3.4, faiss-cpu==1.8.0.post1, FlashRank==0.2.6

"""

from typing import List, Union, Generator, Iterator, Annotated, Sequence
from pydantic import BaseModel
import logging
import json

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage, FunctionMessage


# Import the main logic from jared.py
from jared import start_chat_session, State, MessageDict

class Pipeline:
    class Valves(BaseModel):
        pass

    def __init__(self):
        self.name = "Pipeline Example"
        self.state = State(
            meta_prompt=[],
            conversation_history=[],
            requirements_gathering=[],
            expert_plan="",
            expert_research=[],
            expert_writing="",
            user_input=[],
            previous_search_queries=[],
            router_decision=False,
            chat_limit=0,
            chat_finished=False,
            recursion_limit=5
        )
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger(__name__)

    async def on_startup(self):
        # This function is called when the server is started.
        self.logger.debug("Pipeline startup")
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        self.logger.debug("Pipeline shutdown")
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        pass

    async def inlet(self, body: dict, user: dict) -> dict:
        # This function is called before the OpenAI API request is made. You can modify the form data before it is sent to the OpenAI API.
        self.logger.debug(f"inlet: {body}, user: {user}")
        return body

    async def outlet(self, body: dict, user: dict) -> dict:
        # This function is called after the OpenAI API response is completed. You can modify the messages after they are received from the OpenAI API.
        self.logger.debug(f"outlet: {body}, user: {user}")
        return body

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        self.logger.debug(f"pipe: user_message={user_message}, model_id={model_id}, messages={messages}, body={body}")

        # Prepare the input for the chatbot
        formatted_user_input = self.prepare_pipeline_input(messages)

        # Start the chat session and get the response
        response = start_chat_session(user_message, self.state)
        
        self.logger.debug(f"Chatbot response: {response}")
        return response

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