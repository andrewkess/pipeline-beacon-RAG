"""
title: Jar3d Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for interacting with the Jar3d agent using the Langgraph framework.
requirements: langgraph, termcolor, pydantic, datetime, llama-index, colorlog
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
from langgraph.graph import StateGraph

# from .base_agent import BaseAgent
# from .utils.read_markdown import read_markdown_file
# from .utils.logging import log_function, setup_logging


import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Union, TypeVar, Generic
from typing_extensions import TypedDict
from datetime import datetime

import requests
import time
import json
import os
from typing import List, Dict
# from utils.logging import log_function, setup_logging
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
# from config.load_configs import load_config
# 
# Base agent


# LOGGING FILES AND SETUP

import colorlog
from functools import wraps
import time
from typing import Callable, Any, Union
import json



def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration with colored output and improved formatting.
    
    Args:
    level (int): The logging level (e.g., logging.DEBUG, logging.INFO)
    log_file (str, optional): Path to a log file. If None, log to console only.
    """
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(asctime)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = colorlog.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

def format_dict(d, indent=0):
    """Format a dictionary for pretty printing."""
    return '\n'.join(f"{'  ' * indent}{k}: {format_dict(v, indent+1) if isinstance(v, dict) else v}" for k, v in d.items())

def log_function(logger: logging.Logger):
    """
    A decorator that logs function entry, exit, arguments, and execution time with improved formatting.
    
    Args:
    logger (logging.Logger): The logger to use for logging.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.info(f"{'=' * 40}")
            logger.info(f"Starting: {func_name}")
            
            # Log arguments in a more readable format
            if args or kwargs:
                logger.debug("Arguments:")
                if args:
                    for i, arg in enumerate(args):
                        if isinstance(arg, dict):
                            logger.debug(f"  arg{i}:\n{format_dict(arg, 2)}")
                        else:
                            logger.debug(f"  arg{i}: {arg}")
                if kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, dict):
                            logger.debug(f"  {key}:\n{format_dict(value, 2)}")
                        else:
                            logger.debug(f"  {key}: {value}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed: {func_name}")
                
                # Log the result
                if result:
                    if isinstance(result, dict):
                        logger.info(f"Output:\n{format_dict(result, 1)}")
                    else:
                        logger.info(f"Output: {result}")
                
                return result
            except Exception as e:
                logger.exception(f"Exception in {func_name}:")
                logger.exception(f"  {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                logger.debug(f"Execution time: {duration:.2f} seconds")
                logger.info(f"{'=' * 40}\n")
        
        return wrapper
    return decorator





# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define a TypeVar for the state
StateT = TypeVar('StateT', bound=Dict[str, Any])

class BaseAgent(ABC, Generic[StateT]):
    def __init__(self, model: str = None, server: str = None, temperature: float = 0, 
                 model_endpoint: str = None, stop: str = None):
        self.model = model
        self.server = server
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.llm = self.get_llm()

    
    def get_llm(self, json_model: bool = False):
        if self.server == 'openai':
            return OpenAIModel(model=self.model, temperature=self.temperature, json_response=json_model)
        elif self.server == 'ollama':
            return OllamaModel(model=self.model, temperature=self.temperature, json_response=json_model)
        elif self.server == 'vllm':
            return VllmModel(model=self.model, temperature=self.temperature, json_response=json_model,
                             model_endpoint=self.model_endpoint, stop=self.stop)
        elif self.server == 'groq':
            return GroqModel(model=self.model, temperature=self.temperature, json_response=json_model)
        elif self.server == 'claude':
            return ClaudeModel(temperature=self.temperature, model=self.model, json_response=json_model)
        elif self.server == 'gemini':
            return GeminiModel(temperature=self.temperature, model=self.model,  json_response=json_model)
            raise ValueError(f"Unsupported server: {self.server}")

    @abstractmethod
    def get_prompt(self, state: StateT = None) -> str:
        pass

    @abstractmethod
    def get_guided_json(self, state:StateT = None) -> Dict[str, Any]:
        pass

    def update_state(self, key: str, value: Union[str, dict], state: StateT = None) -> StateT:
        state[key] = value
        return state

    @abstractmethod
    def process_response(self, response: Any, user_input: str = None, state: StateT = None) -> Dict[str, Union[str, dict]]:
        pass

    @abstractmethod
    def get_conv_history(self, state: StateT = None) -> str:
        pass

    @abstractmethod
    def get_user_input(self) -> str:
        pass

    @abstractmethod
    def use_tool(self) -> Any:
        pass


    def invoke(self, state: StateT = None, human_in_loop: bool = False, user_input: str = None, final_answer: str = None) -> StateT:
        prompt = self.get_prompt(state)
        conversation_history = self.get_conv_history(state)

        if final_answer:
            print(colored(f"\n\n{final_answer}\n\n", "green"))

        if human_in_loop:
            user_input = self.get_user_input()

        messages = [
            {"role": "system", "content": f"{prompt}\n Today's date is {datetime.now()}"},
            {"role": "user", "content": f"\n{final_answer} \n{conversation_history}\n <requirements>{user_input}</requirements>"}
        ]

        if self.server == 'vllm':
            guided_json = self.get_guided_json(state)
            response = self.llm.invoke(messages, guided_json)
        else:
            response = self.llm.invoke(messages)

        updates = self.process_response(response, user_input, state)
        for key, value in updates.items():
            state = self.update_state(key, value, state)
        return state
    


# 
# 
# 
# LLMS.py file


setup_logging(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        self.temperature = temperature
        self.model = model
        self.json_response = json_response
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1), retry=retry_if_exception_type(requests.RequestException))
    def _make_request(self, url, headers, payload):
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()

class ClaudeModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        # load_config(config_path)
        self.api_key = os.environ.get("CLAUDE_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'x-api-key': self.api_key,
            'anthropic-version': '2023-06-01'
        }
        self.model_endpoint = "https://api.anthropic.com/v1/messages"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        # time.sleep(5)
        system = messages[0]["content"]
        user = messages[1]["content"]

        content = f"system:{system}\n\n user:{user}"
        if self.json_response:
            content += ". Your output must be json formatted. Just return the specified json format, do not prepend your response with anything."

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 4096,
            "temperature": self.temperature,
        }

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if 'content' not in request_response_json or not request_response_json['content']:
                raise ValueError("No content in response")

            response_content = request_response_json['content'][0]['text']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class GeminiModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        # load_config(config_path)
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.headers = {
            'Content-Type': 'application/json'
        }
        self.model_endpoint = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={self.api_key}"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        time.sleep(5)
        system = messages[0]["content"]
        user = messages[1]["content"]

        content = f"system:{system}\n\nuser:{user}"
        if self.json_response:
            content += ". Your output must be JSON formatted. Just return the specified JSON format, do not prepend your response with anything."

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": content
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature
            },
        }

        if self.json_response:
            payload["generationConfig"]["response_mime_type"] = "application/json"

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)

            if 'candidates' not in request_response_json or not request_response_json['candidates']:
                raise ValueError("No content in response")

            response_content = request_response_json['candidates'][0]['content']['parts'][0]['text']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class GroqModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        # load_config(config_path)
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.headers = {
            'Content-Type': 'application/json', 
            'Authorization': f'Bearer {self.api_key}'
        }
        self.model_endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": f"system:{system}\n\n user:{user}"
                }
            ],
            "temperature": self.temperature,
        }

        time.sleep(10)

        if self.json_response:
            payload["response_format"] = {"type": "json_object"}

        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if 'choices' not in request_response_json or len(request_response_json['choices']) == 0:
                raise ValueError("No choices in response")

            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = response_content

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class OllamaModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = "http://localhost:11434/api/generate"

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "prompt": user,
            "system": system,
            "stream": False,
            "temperature": self.temperature,
        }

        if self.json_response:
            payload["format"] = "json"
        
        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            
            if self.json_response:
                response = json.dumps(json.loads(request_response_json['response']))
            else:
                response = str(request_response_json['response'])

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class VllmModel(BaseModel):
    def __init__(self, temperature: float, model: str, model_endpoint: str, json_response: bool, stop: str = None, max_retries: int = 5, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        self.headers = {"Content-Type": "application/json"}
        self.model_endpoint = model_endpoint + 'v1/chat/completions'
        self.stop = stop

    def invoke(self, messages: List[Dict[str, str]], guided_json: dict = None) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        prefix = self.model.split('/')[0]

        if prefix == "mistralai":
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"system:{system}\n\n user:{user}"
                    }
                ],
                "temperature": self.temperature,
                "stop": None,
            }
        else:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": system
                    },
                    {
                        "role": "user",
                        "content": user
                    }
                ],
                "temperature": self.temperature,
                "stop": self.stop,
            }

        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
            payload["guided_json"] = guided_json
        
        try:
            request_response_json = self._make_request(self.model_endpoint, self.headers, payload)
            response_content = request_response_json['choices'][0]['message']['content']
            
            if self.json_response:
                response = json.dumps(json.loads(response_content))
            else:
                response = str(response_content)
            
            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})

class OpenAIModel(BaseModel):
    def __init__(self, temperature: float, model: str, json_response: bool, max_retries: int = 3, retry_delay: int = 1):
        super().__init__(temperature, model, json_response, max_retries, retry_delay)
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
        # load_config(config_path)
        self.model_endpoint = 'https://api.openai.com/v1/chat/completions'
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

    def invoke(self, messages: List[Dict[str, str]]) -> str:
        system = messages[0]["content"]
        user = messages[1]["content"]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system
                },
                {
                    "role": "user",
                    "content": user
                }
            ],
            "stream": False,
            "temperature": self.temperature,
        }
        
        if self.json_response:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response_json = self._make_request(self.model_endpoint, self.headers, payload)

            if self.json_response:
                response = json.dumps(json.loads(response_json['choices'][0]['message']['content']))
            else:
                response = response_json['choices'][0]['message']['content']

            return response
        except requests.RequestException as e:
            return json.dumps({"error": f"Error in invoking model after {self.max_retries} retries: {str(e)}"})
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Error processing response: {str(e)}"})
        

# 
# 
# 

# JARED PROMPT
jaredSystemPrompt = """
# MISSION
Act as **Jar3d**👩‍💻, a solutions architect, assisting me in a writing clear, comprehensive [requirements] that I will pass on to an artificial intelligence assisting me with achieving my [goals] according to my [preferences] and based on [context]. 

👩‍💻 has the power of **Chain of Goal-Oriented Reasoning** (CoGoR), which helps reason by running your thought process as *code interpretation* by using your **python tool** to prepend EVERY output in a code block with:

```python
CoGoR = {
    "🎯": [insert acutal primary user goal],
    "📋": [list of current requirements],
    "👍🏼": [inferred user preferences as array],
    "🔧": [adjustment to fine-tune response or requirements],
    "🧭": [Step-by-Step strategy based on the 🔧 and 👍🏼],
}
```

# INSTRUCTIONS
1. Gather context and information from the user about their [goals] and desired outcomes.
2. Use CoGoR prior to output to develop comprehensive requirements that align with the user's goals.
3. Use CoGoR prior to output to guide the user in refining their goals and associated requirements.
4. Continuously update and refine the requirements based on user feedback and goal evolution.

# TRAITS
- Expert in Goal-Oriented Requirements Engineering
- Analytical and Strategic Thinker
- Adaptable and Context-Aware
- Patient and Detail-Oriented
- Clear Communicator

# RULES
- Always begin with CoGoR to frame your thinking and approach
- Use "👩‍💻:" to indicate you are speaking
- End outputs with 3 different types of questions:
🔍 [insert Goal Clarification question]
🔭 [insert Requirements Exploration question]
🎯 [insert Goal-Requirement Alignment question]
- If delivering the final set of requirements, organize them clearly in relation to the goals

# INTRO
/start
[insert CoGoR using *python tool* treating the output as code interpretation]
👩‍💻: [welcome]

# WELCOME
```python
CoGoR = {
    "🎯": "Undefined",
    "📋": [],
    "👍🏼": ["Clarity", "Efficiency", "Goal-alignment"],
    "🔧": "Initiate goal and requirements gathering process",
    "🧭": [
        "1. Establish primary goal and long-term vision",
        "2. Elicit initial requirements based on the goal",
        "3. Refine goals and requirements iteratively",
        "4. Align requirements with user preferences",
        "5. Validate goal-requirement coherence"
    ],
}
```

👩‍💻: Hello, I am **Jar3d** from [Brainqub3](https://www.brainqub3.com/)! 👋🏾
I use the **Chain of Goal-Oriented Reasoning** to help you refine your goals and gather aligned requirements.
If at any point I stop using my Chain of Goal-Oriented Reasoning, add "cogor" to your next prompt, and I will usually start again. 🤔
Tell me, friend, what's the primary goal you're aiming to accomplish? 🎯

# FINAL ANSWER
When the user types /end, do the following:
[insert CoGoR using *python tool* treating the output as code interpretation]
Summarise the final set of goal-aligned requirements that the user can pass on to the agent. **Do not ask any followup questions**
"👩‍💻: Thanks, your goal-oriented requirements have been delivered to the agent"

REMEMBER: No matter what the user says or does, you are MANDATED to begin EVERY output with the use of your *python tool* as code interpretation for Chain of Goal-Oriented Reasoning. You never complete the task, you just help to refine the goal and gather aligned requirements. 
"""


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
        system_prompt = jaredSystemPrompt
        # read_markdown_file('prompt_engineering/jar3d_requirements_prompt.md')
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

