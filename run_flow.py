import asyncio
import time

import asyncio
import time
from typing import Any, Dict

import asyncio
import time
from typing import Any, Dict

from app.agent.manus import Manus
from app.flow.flow_factory import FlowFactory, FlowType
from app.logger import logger
from app.communication.blackboard import Blackboard
from app.tool.registry import ToolRegistry
from app.tool.base import BaseTool, ToolResult
from app.planning.planner import HierarchicalPlanner
from app.memory.vector_store import VectorStore
from app.tool.ask_human import AskHuman, AskHumanInput # Import AskHuman tool
from app.agent.base import BaseAgent # Import BaseAgent for custom agent
from app.communication.notifications import HUMAN_ATTENTION_FLAGS_KEY # Key for attention flags

# Define a DummyTool for testing registry functionality
class DummyTool(BaseTool):
    name: str = "dummy_tool"
    description: str = "A simple dummy tool for testing the registry."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "A message to echo."}
        },
        "required": ["message"],
    }

    async def execute(self, **kwargs) -> ToolResult:
        message = kwargs.get("message", "No message provided")
        output = f"DummyTool executed with message: {message}"
        logger.info(output)
        return ToolResult(output=output)


async def run_flow():
    # Initialize Blackboard
    blackboard = Blackboard()

    # Initialize ToolRegistry
    tool_registry = ToolRegistry()

    # Create and register a dummy tool (optional, good for testing)
    dummy_tool_instance = DummyTool()
    tool_registry.register_tool(dummy_tool_instance)
    logger.info(f"Registered tool: {dummy_tool_instance.name} with registry.")

    # Initialize HierarchicalPlanner
    planner = HierarchicalPlanner() 
    logger.info("HierarchicalPlanner initialized.")

    # Initialize VectorStore
    vector_store = VectorStore() 
    logger.info(f"VectorStore initialized with embedding type: {vector_store.get_embedding_type()}.")

    # Create an instance of AskHuman tool and register it
    ask_human_tool = AskHuman()
    tool_registry.register_tool(ask_human_tool)
    logger.info(f"Registered tool: {ask_human_tool.name} with registry.")


    # --- Define a simple InteractiveAgent for demonstration ---
    class InteractiveAgent(BaseAgent):
        name: str = "InteractiveDemoAgent"
        description: str = "An agent to demonstrate AskHuman and flagging."

        async def step(self) -> str:
            """Demonstrates using AskHuman and flagging."""
            if not self.tool_registry:
                return "Tool registry not available."
            
            ask_tool = self.tool_registry.get_tool("ask_human")
            if not ask_tool or not isinstance(ask_tool, AskHuman):
                await self.flag_for_human_attention("AskHuman tool not found in registry!", urgency=5)
                return "AskHuman tool not found."

            # 1. Demonstrate AskHuman with options
            options_input = AskHumanInput(
                query="What should be the next course of action?",
                options=["Proceed with caution", "Gather more data", "Halt and await further instructions"],
                expected_type="text" # The choice will be text
            )
            logger.info(f"{self.name}: Asking human for choice...")
            choice_result: ToolResult = await ask_tool(**options_input.model_dump())
            
            if choice_result.output:
                logger.info(f"{self.name}: Human chose: {choice_result.output}")
                await self.store_experience(f"Human advised: {choice_result.output}")
                if choice_result.output == "Halt and await further instructions":
                    await self.flag_for_human_attention(f"Agent {self.name} was instructed to HALT.", urgency=4)
                    self.state = AgentState.FINISHED # Stop the agent
                    return f"Agent halted by human instruction: {choice_result.output}"
            else:
                logger.error(f"{self.name}: Failed to get valid choice from human.")
                await self.flag_for_human_attention("Failed to get valid choice via AskHuman.", urgency=3)
                return "Failed to get human choice."

            # 2. Demonstrate AskHuman for a filepath
            filepath_input = AskHumanInput(query="Please provide the path to the configuration file:", expected_type="filepath")
            logger.info(f"{self.name}: Asking human for a filepath...")
            filepath_result: ToolResult = await ask_tool(**filepath_input.model_dump())

            if filepath_result.output:
                logger.info(f"{self.name}: Human provided filepath: {filepath_result.output}")
                await self.store_experience(f"Human provided filepath: {filepath_result.output}")
            else:
                logger.error(f"{self.name}: Failed to get filepath from human.")
                # Agent might decide to flag this or try a default
                await self.flag_for_human_attention("Failed to get filepath via AskHuman.", urgency=2)


            # 3. Simulate a situation to flag
            if self.current_step > 1 : # Example condition to flag
                 await self.flag_for_human_attention(f"Agent {self.name} has completed {self.current_step} steps and requires a check-in.", urgency=2)


            if self.current_step >= 3: # Let it run for a few steps
                self.state = AgentState.FINISHED
                return "Interactive demonstration finished after a few steps."
            
            return f"Interactive step {self.current_step} completed. Human choice was: {choice_result.output}, filepath was: {filepath_result.output}"

    # --- End InteractiveAgent Definition ---

    # Initialize agents - Choose one or more to run
    # For this demo, let's use the InteractiveAgent
    # manus_agent = Manus() 
    interactive_agent = InteractiveAgent()

    # Set up the chosen agent
    active_agent = interactive_agent # or manus_agent
    active_agent.set_blackboard(blackboard)
    active_agent.set_tool_registry(tool_registry)
    active_agent.set_planner(planner) # Planner might not be used by InteractiveAgent's simple step
    active_agent.set_vector_store(vector_store)
    logger.info(f"Injected dependencies into agent: {active_agent.name}")
    
    agents_to_run = { # The flow factory might expect a dict
        active_agent.name: active_agent,
    }

    try:
        # For InteractiveAgent, the initial prompt might be a general instruction or ignored if its step() is predefined.
        # For Manus or plan-driven agents, this is the high-level goal.
        user_prompt = input(f"Enter your prompt for {active_agent.name} (or press Enter for default behavior): ")
        
        if not user_prompt.strip() and active_agent.name == "InteractiveDemoAgent":
            user_prompt = "Start interactive demo." # Default prompt for demo agent
        elif not user_prompt.strip():
            logger.warning("Empty prompt provided for a non-demo agent. This might lead to unexpected behavior.")
            # return # Or provide a default goal

        # The FlowFactory and Flow execution might be more complex than needed for a single interactive agent demo
        # For simplicity, we can call agent.run() directly if it's just one agent and no complex flow.
        # However, to stick to the existing structure:
        
        # flow = FlowFactory.create_flow(
        #     flow_type=FlowType.PLANNING, # Adjust if another flow type is more suitable
        #     agents=agents_to_run,
        # )
        # logger.warning(f"Processing your request for {active_agent.name}...")
        # result = await asyncio.wait_for(flow.execute(user_prompt), timeout=3600)
        
        # Simplified run for demonstration:
        logger.warning(f"Running agent {active_agent.name} with request: '{user_prompt}'")
        start_time = time.time()
        # The 'run' method in BaseAgent handles planning if a planner is set and a request is given.
        # For InteractiveAgent, its `step` method has the primary demo logic.
        result = await asyncio.wait_for(active_agent.run(request=user_prompt), timeout=3600)
        elapsed_time = time.time() - start_time
        logger.info(f"Agent {active_agent.name} finished in {elapsed_time:.2f} seconds.")
        logger.info(f"Final result/summary:\n{result}")

    except asyncio.TimeoutError:
        logger.error(f"Agent {active_agent.name} timed out after 1 hour")
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user.")
    except Exception as e:
        logger.error(f"Error during agent execution: {str(e)}", exc_info=True)
    finally:
        logger.info("\n--- Human Attention Flags on Blackboard ---")
        attention_flags = blackboard.get_attention_flags()
        if attention_flags:
            for i, flag_item in enumerate(attention_flags):
                logger.info(f"Flag {i+1}: {flag_item}")
        else:
            logger.info("No human attention flags were raised.")
        logger.info("--- End of Attention Flags ---")


if __name__ == "__main__":
    asyncio.run(run_flow())
