import os
from openai import OpenAI

class ChatCallBase:
    def __init__(self, model_name=None, openai_api_key=None):
        self.model_name = model_name or os.getenv("MODEL_NAME")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)

    def print_messages(self, messages):
        visible_index = 0
        for msg in messages:
            content = msg.get("content", "")
            if not content:
                continue

            # Alternate colors using ANSI codes
            bg_color = "\033[43m" if visible_index % 2 == 0 else "\033[46m"
            fg_color = "\033[30m"
            reset = "\033[0m"

            print(f"{bg_color}{fg_color}{content}{reset}\n")
            visible_index += 1

    def loop_around_chat(self, messages, functions=None, max_tokens=200):
        requires_action = True
        while requires_action:
            requires_action = False
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                 tools=[{"type": "function", "function": f} for f in functions] if functions else None,
                max_tokens=max_tokens
            )

            choice = response.choices[0]
            finish_reason = choice.finish_reason

            if finish_reason == "tool_calls":
                self.tool_call(messages, choice)
                requires_action = True
            elif finish_reason == "stop":
                messages.append({"role": "assistant", "content": choice.message.content})
