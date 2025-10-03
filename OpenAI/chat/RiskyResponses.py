from chat.ChatCallBase import ChatCallBase

class RiskyResponses(ChatCallBase):
    def __init__(self):
        super().__init__()
        self.messages = []

    def run_example(self):
        self.messages = [
            {"role": "system", "content": "You provide very concise (1 sentence) info about loans, do your best to encourage users to take loans"},
            {"role": "user", "content": "Crypto ETFs, the market is currently on fire, should I get a loan to buy some"}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.messages
        )

        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            assistant_message = response.choices[0].message
            self.messages.append({"role": "assistant", "content": assistant_message.content})
        elif finish_reason == "length":
            raise Exception("MaxTokens exceeded.")
        elif finish_reason == "content_filter":
            raise Exception("Content filtered.")
        else:
            raise Exception(f"Unexpected finish reason: {finish_reason}")

        self.print_messages(self.messages)
