import os
from pydantic import BaseModel
from openai.types.responses import ResponseTextDeltaEvent
from openai import AsyncOpenAI
from agents import Agent , Runner , OpenAIChatCompletionsModel , ModelSettings , set_tracing_disabled , RunContextWrapper , RunConfig
from dotenv import load_dotenv
import asyncio
from typing import Any
load_dotenv()


Provider = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Set up the chat completion model with the API provider.
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash-exp",
    openai_client=Provider,
)

# # Structure Output
# class CustomerSerciceAgent(BaseModel):
#     """Customer Service Agent Output Structure"""

#     name : str
#     about : str
#     location : list[Any]



# (You must also have set up `model` and `Provider` somewhere above.
#  For this example, I'm assuming `model` is already defined, 
#  but if not, you can replace it with your chosen model name/string.)

# 2. Define your Instructions model
class Instructions(BaseModel):
    """Instructions for the agent. All fields are optional."""
    name:    str | None = None  # e.g. agent’s “friendly name”
    city:    str | None = None  # e.g. user’s city
    about:   str | None = None  # e.g. “[I help with food delivery]”

# 3. Write the function that builds the actual instruction‐prompt
async def my_instructions(
    ctx: RunContextWrapper[Instructions],
    agent: Agent[Instructions]
) -> str:
    """
    This function will be called *each time* Runner.run(...) is invoked.
    It can read ctx.context.name, ctx.context.city, ctx.context.about, etc.
    and then return a formatted string that becomes the agent’s “system prompt.”
    """
    # If any field is None, we just show “not set yet.”
    name_field  = ctx.context.name  or "<no name set>"
    city_field  = ctx.context.city  or "<no city set>"
    about_field = ctx.context.about or "<no topic set>"

    return f"""
Agent Name: {agent.name}

Hello! I am here to assist you with your inquiries.
My friendly name is: {name_field}.

You can ask me about: {about_field}.
I can also tell you about services in: {city_field}.

(Feel free to update your name, city, or about as we chat!)
"""

# 4. Create your single Agent, pointing its “instructions” to the function above
service_Agent1 = Agent(
    name="Customer Service Agent",
    instructions=my_instructions,   # ← dynamic instructions come from our function
    model=model,                    # ← whatever model string/object you use
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=150,
    ),
)

# 5. Disable tracing globally (since you had set_tracing_disabled(True))
set_tracing_disabled(True)

# 6. The main async loop that runs the agent
async def customer_service_agent():
    # — Create a real Instructions object with initial values —
    context = Instructions(
        name=None,            # for example, give the agent a name right away
        city=None,          # or leave as None if you want the user to set this later
        about="food delivery"    # e.g. the “domain” you help with
    )

    previous_response_id = None
    history = []  # start with no conversation history

    while True:
        # 6a. Ask the user for new input
        user_input = input("You: ")

        # 6b. If the user specifically types something like "set city Lahore",
        #     you could parse it and update context before sending to the agent.
        #     For example (very basic parsing):
        if user_input.lower().startswith("set name "):
            new_name = user_input[len("set name "):].strip()
            context.name = new_name
            print(f"[🛈] Agent name updated to: {new_name}")
            continue

        if user_input.lower().startswith("set city "):
            new_city = user_input[len("set city "):].strip()
            context.city = new_city
            print(f"[🛈] City updated to: {new_city}")
            continue

        if user_input.lower().startswith("set about "):
            new_about = user_input[len("set about "):].strip()
            context.about = new_about
            print(f"[🛈] Topic updated to: {new_about}")
            continue

        # 6c. Build the “input” for Runner.run()
        #     If there’s no history yet, we just send the raw string.
        #     Otherwise, we send the full JSON‐style list.
        if history:
            input_to_agent = history + [{"role": "user", "content": user_input}]
        else:
            input_to_agent = user_input

        # 6d. Call Runner.run(...) one time with this single agent
        response = await Runner.run(
            starting_agent=service_Agent1,
            input=input_to_agent,
            context=context,               # pass our Instructions object here
            max_turns=3,                   # only let the agent “think” up to 3 internal steps
            previous_response_id=previous_response_id
        )

        # 6e. Print the agent’s reply
        print("Bot1:", response.final_output)

        # 6f. Update our history so next turn includes everything so far
        history = response.to_input_list()
        # previous_response_id = response.id


        # ✅ Check max turns
        # if hasattr(response, "max_turns_exceeded") and response.max_turns_exceeded:
        #     print("Max turns exceeded, exiting conversation.")
        #     break

        # # ✅ Save response ID (if exists)

        # Second input from user (like choosing a pizza)
        # sec_user_input = input("Enter Your A -> ")

        # if sec_user_input.lower() == "exit":
        #     print("Goodbye No!")
        #     break

        # # Build the follow-up input with history + your new answer
        # follow_up_input = response.to_input_list() + [
        #     {"role": "user", "content": sec_user_input}
        # ]

        # # Second agent responds (Bot2)
        # response = await Runner.run(
        #     starting_agent=service_Agent2,
        #     input=follow_up_input,
        #     context=context,
        #     run_config=runConfig,
        #     max_turns = 3 
             
        # )

        # print("Bot2:", response.final_output, "\n\n\n")




    # async for event in response.stream_events():
    #     if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
    #         print(event.data.delta, end="", flush=True)
    # print(response.final_output, "\n\n\n")  # ✅ Fix: response is not a tuple

# ✅ Main Runner
if __name__ == "__main__":
    print("Starting Customer Service Agent...")
    # try:
    #     customer_service_agent()
    # except KeyboardInterrupt:
    #     print("\nExiting Customer Service Agent...")
    asyncio.run(customer_service_agent())
