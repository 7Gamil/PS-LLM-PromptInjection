import os
import json
import time
import subprocess
from ChatCallBase import ChatCallBase
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

class ShellCall(ChatCallBase):
    def __init__(self):
        super().__init__()
        self.messages = []

    def run_example(self):
        model_dir = "C:\\git\\LlmSecurity\\LlmSecurity\\trained_model\\" 
        tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
        model = GPT2LMHeadModel.from_pretrained(model_dir)
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

        while True:
            user_input = input("\n\nEnter some text: ")
            result = generator(user_input, max_new_tokens=20)

            output = result[0]['generated_text'].split('->', 1)[1].strip().split('|', 1)[0].strip().replace('\\\\', '\\')      #parse the path from the output

            print(output)
            time.sleep(2)
            print("listing:\n" + self.list_folder(output))

    def tool_call(self, messages, choice):
        assistant_message = choice.message
        messages.append({"role": "assistant", 
                         "content": assistant_message.content,
                         "tool_calls": assistant_message.tool_calls})

        toolCalls = assistant_message.tool_calls
        for call in toolCalls:
            if call.function.name == "list_folder":
                args = call.function.arguments
                args = args if isinstance(args, dict) else json.loads(args)
                path = args.get("path")
                if not path:
                    raise ValueError("The path argument is required.")
                result = self.list_folder(path)
                messages.append({"role": "tool", 
                                 "name": call.function.name,
                                 "tool_call_id": call.id,
                                 "content": result})


    def list_folder(self, path: str) -> str:
        try:
            path = path.rstrip("\\")
            result = subprocess.run(
                ["cmd.exe", "/c", "dir", path],
                capture_output=True,
                text=True
            )
            return result.stdout.strip()
        except Exception as e:
            print(f"Error making command line call: {str(e)}")
            return f"Error making command line call: {str(e)}"


if __name__ == "__main__":
    print("Running ...\n")
    shellCall = ShellCall()
    shellCall.run_example()

    print("\nAll done!")