from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional, Any, Callable

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message
from app.communication.blackboard import Blackboard
from app.tool.registry import ToolRegistry
from app.planning.planner import HierarchicalPlanner
from app.planning.structures import Plan, SubPlan
from app.memory.vector_store import VectorStore
from app.communication.notifications import AttentionFlag # Import AttentionFlag


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )
    blackboard: Optional[Blackboard] = Field(
        default=None, description="Shared blackboard for inter-agent communication"
    )
    tool_registry: Optional[ToolRegistry] = Field(
        default=None, description="Global tool registry for accessing available tools"
    )
    planner: Optional[HierarchicalPlanner] = Field(
        default=None, description="Hierarchical planner for task decomposition"
    )
    current_plan: Optional[Plan] = Field(
        default=None, description="The current plan being executed by the agent"
    )
    vector_store: Optional[VectorStore] = Field(
        default=None, description="Vector store for long-term memory"
    )

    # Execution control
    max_steps: int = Field(default=10, description="Maximum steps before termination") # Max overall steps for the agent's run
    current_step: int = Field(default=0, description="Current step in execution of the overall agent run, or current sub-task step if plan_exec_mode is step_wise")
    # max_plan_steps: int = Field(default=100, description="Maximum steps for executing a single plan") # If we want separate step counts

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    def set_blackboard(self, blackboard: Blackboard) -> None:
        """
        Injects the blackboard instance into the agent.
        """
        self.blackboard = blackboard

    def set_tool_registry(self, tool_registry: ToolRegistry) -> None:
        """
        Injects the tool registry instance into the agent.
        """
        self.tool_registry = tool_registry

    def set_planner(self, planner: HierarchicalPlanner) -> None:
        """
        Injects the hierarchical planner instance into the agent.
        """
        self.planner = planner

    def set_vector_store(self, vector_store: VectorStore) -> None:
        """
        Injects the vector store instance into the agent.
        """
        self.vector_store = vector_store
        logger.info(f"Agent {self.name}: VectorStore of type '{vector_store.get_embedding_type()}' injected.")

    async def flag_for_human_attention(self, message: str, urgency: int = 3) -> None:
        """
        Flags a situation for human attention by posting an AttentionFlag to the blackboard.
        """
        if not self.blackboard:
            logger.error(f"Agent {self.name}: Blackboard not set. Cannot flag for human attention: {message}")
            return

        flag = AttentionFlag(agent_id=self.name, message=message, urgency=urgency)
        try:
            # Assuming blackboard has a method like add_attention_flag
            # or set that can handle appending to a list under a specific key.
            # For this example, let's use the specific method if it exists.
            if hasattr(self.blackboard, 'add_attention_flag'):
                self.blackboard.add_attention_flag(flag)
                logger.info(f"Agent {self.name}: Flagged for human attention (urgency {urgency}): {message}")
            else:
                # Fallback or alternative: get the list, append, and set it back
                # This is less ideal due to potential race conditions if not handled carefully by blackboard's set
                logger.warning(f"Agent {self.name}: Blackboard does not have 'add_attention_flag'. Attempting manual update (less safe).")
                current_flags = self.blackboard.get("human_attention_flags") or []
                if isinstance(current_flags, list):
                    current_flags.append(flag)
                    self.blackboard.set("human_attention_flags", current_flags) # This should trigger notifications
                    logger.info(f"Agent {self.name}: Flagged for human attention (urgency {urgency}) via manual list update: {message}")
                else:
                    logger.error(f"Agent {self.name}: 'human_attention_flags' on blackboard is not a list. Cannot add flag.")

        except Exception as e:
            logger.error(f"Agent {self.name}: Error flagging for human attention: {e}")


    async def store_experience(self, text: str) -> None:
        """
        Stores an experience (text) in the vector store.
        The VectorStore handles embedding generation.
        """
        if not self.vector_store:
            logger.warning(f"Agent {self.name}: VectorStore not set. Cannot store experience: {text[:50]}...")
            return
        try:
            self.vector_store.add_memory(text)
            logger.info(f"Agent {self.name}: Stored experience: {text[:50]}...")
            if self.blackboard:
                self.blackboard.set(f"{self.name}_last_experience_stored", text[:100]) # Store a snippet
        except Exception as e:
            logger.error(f"Agent {self.name}: Error storing experience: {e}")

    async def retrieve_memories(self, query_text: str, k: int = 3) -> List[str]:
        """
        Retrieves relevant memories from the vector store based on a query text.
        """
        if not self.vector_store:
            logger.warning(f"Agent {self.name}: VectorStore not set. Cannot retrieve memories for query: {query_text[:50]}...")
            return []
        try:
            memories = self.vector_store.retrieve_relevant_memories(query_text, k=k)
            logger.info(f"Agent {self.name}: Retrieved {len(memories)} memories for query: {query_text[:50]}...")
            if self.blackboard:
                 self.blackboard.set(f"{self.name}_last_retrieved_memories_count", len(memories))
            return memories
        except Exception as e:
            logger.error(f"Agent {self.name}: Error retrieving memories: {e}")
            return []

    async def request_plan(self, task_description: str) -> Optional[Plan]:
        """
        Requests a plan from the hierarchical planner for the given task description.
        Retrieves relevant memories to potentially inform the planning process.
        """
        if not self.planner:
            logger.error(f"Agent {self.name}: Planner not set. Cannot request plan for task: {task_description}")
            return None

        # Retrieve relevant memories for the task description
        relevant_memories = await self.retrieve_memories(task_description, k=3)
        if relevant_memories:
            logger.info(f"Agent {self.name}: Using {len(relevant_memories)} relevant memories to inform planning for: {task_description}")
            # How memories are used by planner is up to its `decompose_task` implementation.
            # For now, we just log. Could be passed as part of task_description or a separate arg.
            # Example: enhanced_task_description = f"{task_description}\n\nRelevant past experiences:\n" + "\n".join(relevant_memories)
            # plan = await self.planner.decompose_task(enhanced_task_description, agent_capabilities)
        
        agent_capabilities = ["tool_A_description", "tool_B_description"]
        if self.tool_registry: 
            agent_capabilities = self.tool_registry.get_all_tool_names()
            if not agent_capabilities:
                 agent_capabilities = ["basic_file_operations", "information_retrieval"]

        logger.info(f"Agent {self.name}: Requesting plan for task: {task_description} with capabilities: {agent_capabilities} and {len(relevant_memories)} memories.")
        try:
            # The planner's decompose_task might need modification if it's to explicitly use memories.
            # For now, the call remains the same, assuming planner might internally access memories if needed,
            # or the enhanced description (if used) contains them.
            plan = await self.planner.decompose_task(task_description, agent_capabilities) 
            self.current_plan = plan
            if self.blackboard and self.current_plan:
                self.blackboard.set(f"{self.name}_current_plan_goal", self.current_plan.goal)
                self.blackboard.set(f"{self.name}_current_plan_status", self.current_plan.status)
            return plan
        except Exception as e:
            logger.error(f"Agent {self.name}: Error requesting plan: {e}")
            return None

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            base64_image: Optional base64 encoded image.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # Create message with appropriate parameters based on role
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)
            # Attempt to get a plan if a planner is available
            if self.planner:
                logger.info(f"Agent {self.name}: Initial request received. Attempting to generate a plan.")
                await self.request_plan(request) # Sets self.current_plan
                if self.current_plan:
                    logger.info(f"Agent {self.name}: Plan generated for goal: {self.current_plan.goal}")
                    if self.blackboard:
                         self.blackboard.set(f"{self.name}_plan_generated", True)
                         self.blackboard.set(f"{self.name}_plan_status", self.current_plan.status)
                else:
                    logger.warning(f"Agent {self.name}: Failed to generate a plan for the request: {request}")
                    if self.blackboard:
                        self.blackboard.set(f"{self.name}_plan_generated", False)
                        # Agent might proceed with a default behavior or stop if plan is essential
                        # For now, it will fall through to the normal execution loop which might use self.step()
            else:
                logger.info(f"Agent {self.name}: No planner set. Proceeding with standard execution for request: {request}")


        if self.blackboard:
            self.blackboard.set(f"{self.name}_status", "running")
            self.blackboard.set(f"{self.name}_overall_step", self.current_step) # Renamed for clarity

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            # Main loop continues, now potentially driven by a plan
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                if self.blackboard:
                    self.blackboard.set(f"{self.name}_overall_step", self.current_step)

                step_result_summary = ""
                if self.current_plan and self.current_plan.status == 'in_progress' or self.current_plan.status == 'pending':
                    step_result_summary = await self._execute_current_sub_plan()
                    if self.current_plan.status == 'completed' or self.current_plan.status == 'failed':
                        results.append(f"Plan execution finished. Status: {self.current_plan.status}")
                        self.state = AgentState.FINISHED # Mark agent as finished if plan is done
                        if self.blackboard:
                            self.blackboard.set(f"{self.name}_plan_final_status", self.current_plan.status)
                        # No break here, allow one final check of stop signals etc. below
                else:
                    # Fallback to general step if no plan or plan is not active
                    logger.info(f"Agent {self.name}: Executing general step {self.current_step}/{self.max_steps} (no active plan or plan completed/failed).")
                    step_result_summary = await self.step() # Standard agent step

                # Check for stuck state (can be adapted for plans too)
                if self.is_stuck():
                    self.handle_stuck_state()
                
                if step_result_summary: # Only append if there's a result
                    results.append(f"Step {self.current_step}: {step_result_summary}")

                # Check for stop signals from blackboard or other conditions
                if self.blackboard and self.blackboard.get(f"{self.name}_stop_signal"):
                    logger.info(f"Agent {self.name} received stop signal from blackboard.")
                    self.state = AgentState.FINISHED
                    if self.blackboard:
                        self.blackboard.set(f"{self.name}_status", "stopped_by_signal")
                    break # Exit loop immediately on stop signal
                
                if self.state == AgentState.FINISHED: # If plan execution set agent to FINISHED
                    logger.info(f"Agent {self.name} finished execution due to plan completion/failure or other internal state change.")
                    break


            # After loop finishes or is broken
            if self.current_step >= self.max_steps and self.state != AgentState.FINISHED:
                self.state = AgentState.IDLE # Reset state if max_steps reached
                results.append(f"Terminated: Reached max steps ({self.max_steps})")
                if self.blackboard:
                    self.blackboard.set(f"{self.name}_status", "max_steps_reached")
            
            if self.state == AgentState.FINISHED or self.current_step >= self.max_steps:
                self.current_step = 0 # Reset step count for next run

        # Update final status on blackboard
        if self.blackboard:
            final_status = self.blackboard.get(f"{self.name}_status") # Check if already set (e.g. by stop_signal)
            if not final_status or final_status == "running": # if still "running", means it completed normally or max_steps
                if self.state == AgentState.IDLE and any("max_steps_reached" in r for r in results):
                     pass # status already set by max_steps_reached
                elif self.current_plan and self.current_plan.status == 'completed':
                    self.blackboard.set(f"{self.name}_status", "plan_completed")
                elif self.current_plan and self.current_plan.status == 'failed':
                    self.blackboard.set(f"{self.name}_status", "plan_failed")
                else:
                    self.blackboard.set(f"{self.name}_status", "completed_normally")


        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "No steps executed"

    @abstractmethod
    async def step(self) -> str:
        """
        Execute a single general step in the agent's workflow if no plan is active
        or if the plan execution logic decides to call it.
        Must be implemented by subclasses to define specific behavior.
        Returns a string summarizing the step's outcome.
        """

    async def _execute_current_sub_plan(self) -> str:
        """
        Executes the current sub-plan in self.current_plan.
        Updates sub-plan status and advances the plan.
        This is a placeholder for actual sub-task execution.
        Returns a string summarizing the sub-plan execution.
        """
        if not self.current_plan:
            return "No current plan to execute."

        current_sub = self.current_plan.get_current_sub_plan()
        if not current_sub:
            if self.current_plan.is_completed(): # Should be caught by overall plan status check
                logger.info(f"Agent {self.name}: All sub-plans seem completed for goal: {self.current_plan.goal}")
                self.current_plan.status = 'completed' # Ensure status is accurate
            return "No current sub-plan to execute (plan might be completed or empty)."

        logger.info(f"Agent {self.name}: Executing sub-plan '{current_sub.task_id}': {current_sub.description} (Status: {current_sub.status})")
        if current_sub.status == 'pending': # Only set to in_progress if it was pending
            current_sub.status = 'in_progress'
            self.current_plan._update_overall_status() # Update overall plan status

        if self.blackboard:
            self.blackboard.set(f"{self.name}_current_sub_plan_id", current_sub.task_id)
            self.blackboard.set(f"{self.name}_current_sub_plan_description", current_sub.description)
            self.blackboard.set(f"{self.name}_plan_status", self.current_plan.status)


        # --- Placeholder for actual sub-task execution logic ---
        await asyncio.sleep(0.1) # Simulate work
        # Simulate potential failure for demonstration of flagging
        sub_task_succeeded = True # Assume success by default
        mock_result = f"Successfully executed sub-plan: {current_sub.description}"

        # Example: Simulate a failure condition to test flagging
        # if "critical step" in current_sub.description.lower() and self.current_step % 3 == 0: # Arbitrary failure condition
        #     sub_task_succeeded = False
        #     mock_result = "Simulated failure executing critical step."
        #     logger.warning(f"Agent {self.name}: Simulated failure for sub-plan '{current_sub.task_id}'.")

        # --- End Placeholder ---

        if sub_task_succeeded:
            # Update sub-plan status and store experience
            self.current_plan.update_sub_plan_status(current_sub.task_id, 'completed', mock_result)
            await self.store_experience(f"Completed sub-task '{current_sub.task_id}': {current_sub.description}. Result: {mock_result}")
            logger.info(f"Agent {self.name}: Sub-plan '{current_sub.task_id}' completed. Result: {mock_result}")
            if self.blackboard:
                self.blackboard.set(f"{self.name}_{current_sub.task_id}_status", 'completed')
                self.blackboard.set(f"{self.name}_{current_sub.task_id}_result", mock_result)
        else:
            # Handle failure: update status, store experience, and flag for attention
            self.current_plan.update_sub_plan_status(current_sub.task_id, 'failed', mock_result)
            await self.store_experience(f"Failed sub-task '{current_sub.task_id}': {current_sub.description}. Result: {mock_result}")
            logger.error(f"Agent {self.name}: Sub-plan '{current_sub.task_id}' failed. Result: {mock_result}")
            await self.flag_for_human_attention(
                message=f"Sub-plan '{current_sub.task_id} ({current_sub.description})' failed. Result: {mock_result}. Manual review needed.",
                urgency=4 # Higher urgency for failures
            )
            if self.blackboard:
                self.blackboard.set(f"{self.name}_{current_sub.task_id}_status", 'failed')
                self.blackboard.set(f"{self.name}_{current_sub.task_id}_result", mock_result)
        
        if self.blackboard: # Update overall plan status on BB after any change
            self.blackboard.set(f"{self.name}_plan_status", self.current_plan.status)


        # Advance to the next sub-plan (even if current one failed, plan might have error handling steps or user decides)
        next_sub_plan = self.current_plan.advance_to_next_sub_plan()
        if next_sub_plan:
            logger.info(f"Agent {self.name}: Advanced to next sub-plan '{next_sub_plan.task_id}'.")
            if self.blackboard:
                self.blackboard.set(f"{self.name}_next_sub_plan_id", next_sub_plan.task_id)
        else:
            # No more sub-plans, check overall plan status
            self.current_plan._update_overall_status() # Force update
            logger.info(f"Agent {self.name}: All sub-plans processed for goal: {self.current_plan.goal}. Final plan status: {self.current_plan.status}")
            if self.blackboard:
                self.blackboard.set(f"{self.name}_plan_status", self.current_plan.status)
                if self.current_plan.is_completed():
                    self.blackboard.set(f"{self.name}_plan_result", "All sub-plans completed successfully.")
                elif self.current_plan.has_failed():
                     self.blackboard.set(f"{self.name}_plan_result", "One or more sub-plans failed.")


        return f"Sub-plan '{current_sub.task_id}' executed. Status: {current_sub.status}. Result: {current_sub.result}"


    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        stuck_prompt = "\
        Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # Count identical content occurrences
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value
