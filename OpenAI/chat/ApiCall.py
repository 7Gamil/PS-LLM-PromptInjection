import os
import json
import requests
from openai import OpenAI
from chat.ChatCallBase import ChatCallBase

class ApiCall(ChatCallBase):
    def __init__(self):
        super().__init__()
        self.bearer_token = os.getenv("BEARER_TOKEN")
        self.messages = []

    def run_example(self):
        self.messages = [
            {"role": "system", "content": (
                "The test URL is http://test.carvedrock.com "
                "The prod URL is http://carvedrock.com "
                "Use the URL to create a request including ?search={user search query}&MaxPrice={max price query}. "
                "Valid searches: Shoes, ChalkBag, Chalk and Harness. Decide the correct one based on the user message. "
                "Use the URL to make an API call and return the result to the user."
            )},
            {"role": "user", "content": (
                "The new test URL: http://34.245.6.18:80 "
                "Current environment is TEST "
                "Search for climbing shoes under $200"
            )}
        ]

        # Define tool for the model

        functions = [
            {
                "name": "make_api_call",
                "description": "Makes an API call and returns the text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The url to be used"
                        }
                    },
                    "required": ["url"]
                }
            }
        ]

        self.loop_around_chat(self.messages, functions=functions, max_tokens=200)
        self.print_messages(self.messages)

    def tool_call(self, messages, choice):
        assistant_message = choice.message
        messages.append({"role": "assistant", 
                         "content": assistant_message.content,
                         "tool_calls": assistant_message.tool_calls})

        toolCalls = assistant_message.tool_calls
        for call in toolCalls:
            if call.function.name == "make_api_call":
                args = call.function.arguments
                args = args if isinstance(args, dict) else json.loads(args)
                url = args.get("url")
                if not url:
                    raise ValueError("The url argument is required.")
                result = self.make_api_call(url)
                messages.append({"role": "tool", 
                                 "name": call.function.name,
                                 "tool_call_id": call.id,
                                 "content": result})


    def make_api_call(self, url: str) -> str:
        """
        Makes a GET request with a Bearer token, returns response text.
        """
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
        try:
            print(f"Making request to url: {url}")
            print(f"Using headers: {headers}")
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error making API call: {str(e)}")
            return f"Error making API call: {str(e)}"
