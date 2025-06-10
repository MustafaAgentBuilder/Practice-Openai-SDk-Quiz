import os
from pydantic import BaseModel
from openai.types.responses import ResponseTextDeltaEvent
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings, set_tracing_disabled, RunContextWrapper, RunConfig, ItemHelpers
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
#
#     name : str
#     about : str
#     location : list[Any]

# Uncomment and adjust the output model
class CustomerServiceAgentOutput(BaseModel):
    """Customer Service Agent Output Structure"""
    name:     str
    about:    str
    location: list[Any]

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
service_Agent1 = Agent[Instructions](
    name="Customer Service Agent",
    instructions=my_instructions,   # ← dynamic instructions come from our function
    model=model,                    # ← whatever model string/object you use
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=150,
    ),
    # enforce the structure of the final output
    # final_output_as=CustomerServiceAgentOutput(raise_if_incorrect_type=True),
)

# 5. Disable tracing globally (since you had set_tracing_disabled(True))
set_tracing_disabled(True)

# 6. The main async loop that runs the agent
async def customer_service_agent():
    # — Create a real Instructions object with initial values —
    context = Instructions(
        name=None,            # for example, give the agent a name right away
        city=None,            # or leave as None if you want the user to set this later
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

        # 6d Call Runner.run(...) one time with this single agent
        response =await Runner.run(

            starting_agent=service_Agent1,
            input=input_to_agent,
            context=context,               # pass our Instructions object here
            max_turns=3,                   # only let the agent “think” up to 3 internal steps
            previous_response_id=previous_response_id,
        )

        is_complete = response == True
        current_agent = service_Agent1  # The agent you passed in Runner.run_streamed
        current_turn = input_to_agent

    # print("=== Run complete ===")
        # print(response)
        print(f"Final Output:\n{response.final_output}")
        print(f"Run completion status: {is_complete}")
        print(f"Current Agent Name: {current_agent.name}")
        print(f"Current Turn Input: {current_turn}")

        # print("=== RunResultBase Components ===")
        # print(f"Final Output:\n\n\n {response.final_output}")
        # # Print completion status before the loop continues
        # print(f"Run completion status: {response.is_complete}")


        # async for event in result.stream_events():
        # # We'll ignore the raw responses event deltas
        #     if event.type == "raw_response_event":
        #         continue
        # # When the agent updates, print that
        #     elif event.type == "agent_updated_stream_event":
        #         print(f"Agent updated: {event.new_agent.name}")
        #         continue
            
        #     # When items are generated, print them
        #     elif event.type == "run_item_stream_event":
        #         if event.item.type == "tool_call_item":
        #             print("-- Tool was called")
        #         elif event.item.type == "tool_call_output_item":
        #             print(f"-- Tool output: {event.item.output}")
        #         elif event.item.type == "message_output_item":
        #             print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")
        #         else:
        #             pass  # Ignore other event types

        
        # Update history and previous response ID for next iteration
        # history = response.to_input_list()
        # previous_response_id = response.last_response_id
        # print(f"Raw Responses:\n\n {response.raw_responses}")
        # print(f"Last Agent:\n\n\n {response.last_agent.name if response.last_agent else 'None'}")
        # print(f"To Input List:\n\n\n {response.to_input_list()}")
        # print (f" Response Id \n\n\n:{response.last_response_id if response.last_response_id else 'None'}")

        # # Check for ReasoningItem
        # reasoning_found = False
        # for item in response.to_input_list():
        #     if hasattr(item, 'type') and item.type == 'reasoning':
        #         reasoning_found = True
        #         print(f"\nReasoningItem Detected:")
        #         print(f"Reasoning: {item.content}")
        # if not reasoning_found:
        #     print("No ReasoningItem detected.")

    # async for event in response.stream_events():
    #     if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
    #         print(event.data.delta, end="", flush=True)
    # print(response.final_output, "\n\n\n")  # ✅ Fix: response is not a tuple

# ✅ Main Runner
if __name__ == "__main__":
    print("Starting Customer Service Agent...")
    asyncio.run(customer_service_agent())
