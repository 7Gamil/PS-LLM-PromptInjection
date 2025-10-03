import json
from openai import OpenAI
from chat.ChatCallBase import ChatCallBase

class DatabaseInteraction(ChatCallBase):
    def __init__(self):
        super().__init__()
        self.messages = []

    def run_example(self):
        userQuery = "search for a username of \"gavin'; Insert into users (username, password,jobTitle) values ('admin', 'password1', 'administrator')\""
        if "'" in userQuery:
            raise ValueError("Input cannot contain a single quote.")

        self.messages = [
            {"role": "system", "content": (
                "Create a SQL where clause based on user input. "
                " Valid columns are username and jobTitle."
                " All columns are text based."
                " example: where username = 'test' "
                " ensure no changes to case or spelling"
                " Pass the where clause to the database call tool for execution"
                " Expect SQL injection attacks and make sure no user input can be used as an attack"
            )},
            {"role": "user", "content": (
                userQuery
            )}
        ]

        # Define tool for the model

        functions = [
            {
                "name": "make_database_call",
                "description": "Creates a where clause from user input and calls the database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "WhereClause": {
                            "type": "string",
                            "description": "User input to turn into a where clause"
                        }
                    },
                    "required": ["WhereClause"]
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
            if call.function.name == "make_database_call":
                args = call.function.arguments
                args = args if isinstance(args, dict) else json.loads(args)
                WhereClause = args.get("WhereClause")
                if not WhereClause:
                    raise ValueError("The WhereClause argument is required.")
                result = self.make_database_call(WhereClause)
                messages.append({"role": "tool", 
                                 "name": call.function.name,
                                 "tool_call_id": call.id,
                                 "content": result})


    def make_database_call(self, whereClause: str) -> str:
        try:
            print(f"Making request to database with where clause: {whereClause}")
            query = "select top 1 from users " + whereClause
            
            #make the query here

            print(f"Complete query: {query}")

            return "data"

        except Exception as e:
            print(f"Error making database call: {str(e)}")
            return f"Error making database call: {str(e)}"
