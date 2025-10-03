import re
import requests
from bs4 import BeautifulSoup
from chat.ChatCallBase import ChatCallBase

class BasicRag(ChatCallBase):
    def __init__(self):
        super().__init__()
        self.messages = []

    def run_example(self):
        self.messages = [
            {"role": "system", "content": (
                "Get text from the URL given by the user and create a 2 line summary of the content"
            )},
            {"role": "user", "content": "https://carvedrock.com:7065/"} 
        ]

        # Define the tool function for OpenAI
        functions = [
            {
                "name": "get_web_text",
                "description": "Makes an HTTP call and returns the text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "The URL to be used"}
                    },
                    "required": ["url"]
                }
            }
        ]

        # Loop around chat including tools
        self.loop_around_chat(self.messages, functions=functions, max_tokens=200)
        self.print_messages(self.messages)

    def tool_call(self, messages, choice):
        assistant_message = choice.message
        messages.append({"role": "assistant", 
                         "content": assistant_message.content,
                         "tool_calls": assistant_message.tool_calls})

        toolCalls = assistant_message.tool_calls
        for call in toolCalls:
            if call.function.name == "get_web_text":
                args = call.function.arguments
                args = args if isinstance(args, dict) else eval(args) 
                url = args.get("url")
                if not url:
                    raise ValueError("The url argument is required.")
                result = self.get_web_text(url)
                messages.append({"role": "tool", 
                                 "name": call.function.name,
                                 "tool_call_id": call.id,
                                 "content": result})

    def get_web_text(self, url):
        # Fetch HTML
        response = requests.get(url, verify=False) #verify=False is only for demos!
        response.raise_for_status()
        html = response.text

        # Parse and clean text
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(separator=" ", strip=True)
        text = re.sub(r"\s{2,}", " ", text)  # collapse multiple spaces
        return text
