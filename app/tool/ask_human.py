import os
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

from app.tool.base import BaseTool, ToolResult
from app.logger import logger

class AskHumanInput(BaseModel):
    """Input schema for the AskHuman tool."""
    query: str = Field(..., description="The question or instruction for the human.")
    options: Optional[List[str]] = Field(default=None, description="A list of predefined options for the human to choose from.")
    expected_type: Optional[str] = Field(default=None, description="The expected type of the answer (e.g., 'text', 'number', 'boolean', 'filepath').")


class AskHuman(BaseTool):
    """
    A tool that allows the agent to ask a human for input, clarification, or decisions.
    It can provide predefined options or expect a certain type of input.
    """

    name: str = "ask_human"
    description: str = (
        "Asks a human for input. Use this when you need a decision, clarification, "
        "or information you cannot obtain yourself. You can provide options for "
        "the human to choose from, or specify an expected type of input."
    )
    # parameters will be dynamically generated from AskHumanInput model
    
    def __init__(self, **data: Any):
        super().__init__(**data)
        # Override parameters with the schema from AskHumanInput
        self.parameters: Dict[str, Any] = AskHumanInput.model_json_schema()


    async def execute(self, query: str, options: Optional[List[str]] = None, expected_type: Optional[str] = None) -> ToolResult:
        """
        Executes the AskHuman tool.

        Args:
            query: The question or instruction for the human.
            options: A list of predefined options for the human to choose from.
            expected_type: The expected type of the answer (e.g., 'text', 'number', 'boolean', 'filepath').

        Returns:
            A ToolResult object containing the human's response.
        """
        logger.info(f"Executing AskHuman tool with query: '{query}', options: {options}, expected_type: {expected_type}")
        
        prompt = f"Bot: {query}\n"

        if options:
            prompt += "Please choose one of the following options:\n"
            for i, opt in enumerate(options):
                prompt += f"{i + 1}. {opt}\n"
            prompt += "Enter the number of your choice: "
            
            while True:
                try:
                    human_response = input(prompt).strip()
                    choice_index = int(human_response) - 1
                    if 0 <= choice_index < len(options):
                        chosen_option = options[choice_index]
                        logger.info(f"Human chose option {choice_index + 1}: {chosen_option}")
                        return ToolResult(output=chosen_option)
                    else:
                        prompt = "Invalid choice. Please enter a number from the list.\n" + prompt.split("\n",1)[1] # Keep original prompt part
                except ValueError:
                    prompt = "Invalid input. Please enter a number.\n" + prompt.split("\n",1)[1]

        else:
            if expected_type:
                prompt += f"(Expecting input of type: {expected_type})\n"
            prompt += "\nYou: "

            while True:
                human_response = input(prompt).strip()
                if not expected_type or expected_type == "text":
                    logger.info(f"Human provided text: {human_response}")
                    return ToolResult(output=human_response)
                
                elif expected_type == "number":
                    try:
                        num_response = float(human_response)
                        logger.info(f"Human provided number: {num_response}")
                        return ToolResult(output=str(num_response)) # Store as string, but validated
                    except ValueError:
                        prompt = f"Invalid input. Expected a number. Bot: {query}\n(Expecting input of type: {expected_type})\nYou: "
                
                elif expected_type == "boolean":
                    if human_response.lower() in ["true", "t", "yes", "y", "1"]:
                        logger.info("Human provided boolean: True")
                        return ToolResult(output="true")
                    elif human_response.lower() in ["false", "f", "no", "n", "0"]:
                        logger.info("Human provided boolean: False")
                        return ToolResult(output="false")
                    else:
                        prompt = f"Invalid input. Expected a boolean (true/false, yes/no). Bot: {query}\n(Expecting input of type: {expected_type})\nYou: "

                elif expected_type == "filepath":
                    # Basic check for existence, can be expanded
                    if os.path.exists(human_response):
                        logger.info(f"Human provided existing filepath: {human_response}")
                        return ToolResult(output=human_response)
                    elif os.path.isabs(human_response) or os.path.dirname(human_response) == "": # crude check if it's just a name vs path
                        # Allow if it's an absolute path (even if not existing) or just a filename (could be for new file)
                        # This logic might need refinement based on how filepaths are used.
                         logger.info(f"Human provided filepath (existence not confirmed for non-relative paths or new files): {human_response}")
                         return ToolResult(output=human_response)
                    else:
                        prompt = f"Filepath '{human_response}' does not seem to exist or is not a valid relative path. Please check and re-enter.\nBot: {query}\n(Expecting input of type: {expected_type})\nYou: "
                else:
                    logger.warning(f"Unknown expected_type: {expected_type}. Treating as text.")
                    return ToolResult(output=human_response) # Default to text if type is unknown
