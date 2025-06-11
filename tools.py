@function_tool(
    name_override="add_numbers",
    description_override="Adds two numbers together.",
    docstring_style = "google",
    use_docstring_info = True,
    failure_error_function = default_tool_error_function,  
    strict_mode = True,  

)
async def add_numbers(a: int, b: int) -> int:
    """
    Adds two numbers together.
    
    Args:
        a (int): The first number.
        b (int): The second number.
        
    Returns:
        int: The sum of the two numbers.
    """
    return a + b

# 4. Create your single Agent, pointing its “instructions” to the function above
service_Agent1 = Agent[Instructions](
    name="Customer Service Agent",
    instructions=my_instructions,   # ← dynamic instructions come from our function
    model=model,  

    tool_use_behavior=StopAtTools(
        stop_at_tool_names=["add_numbers"],            # and reset tool choice after using it
    ),
    reset_tool_choice=True,
    tools= [add_numbers],                                    # ← whatever model string/object you use
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=150,
        tool_choice= "auto",
        parallel_tool_calls= None,  # allow parallel tool calls
        truncation = "disabled",

    ),
    # enforce the structure of the final output
    # final_output_as=CustomerServiceAgentOutput(raise_if_incorrect_type=True),
)
