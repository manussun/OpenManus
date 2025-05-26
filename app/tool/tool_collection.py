"""Collection classes for managing multiple tools."""
from typing import Any, Dict, List, Optional, Set
from app.tool.registry import ToolRegistry # Import ToolRegistry
from app.exceptions import ToolError
from app.logger import logger
from app.tool.base import BaseTool, ToolFailure, ToolResult


class ToolCollection:
    """
    A collection of tools, potentially drawing from a central ToolRegistry
    and/or a list of explicitly provided tools.
    """

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *tools: BaseTool, tool_registry: Optional[ToolRegistry] = None):
        """
        Initializes the ToolCollection.

        Args:
            *tools: A sequence of BaseTool instances to be included directly.
            tool_registry: An optional ToolRegistry instance. If provided, tools 
                           from the registry will be accessible, with explicitly 
                           passed tools taking precedence in case of name conflicts.
        """
        self.tool_registry: Optional[ToolRegistry] = tool_registry
        self.tool_map: Dict[str, BaseTool] = {}

        # Populate from registry first (if provided)
        if self.tool_registry:
            for tool in self.tool_registry.list_tools():
                self.tool_map[tool.name] = tool
        
        # Add explicitly passed tools, potentially overwriting registry tools
        # This also populates self.tools with a unique list of tools for iteration
        # respecting the precedence (explicitly passed > registry)
        # and also ensures self.tools doesn't have duplicates if a tool is in both.
        
        # Store explicitly passed tools separately to manage them for add/remove if needed
        self._explicit_tools: List[BaseTool] = list(tools) 

        for tool in tools:
            self.tool_map[tool.name] = tool # Explicit tools take precedence

        # self.tools should reflect the combined unique set, respecting precedence
        # The easiest way is to use the values from the final tool_map
        self.tools: tuple[BaseTool, ...] = tuple(self.tool_map.values())


    def __iter__(self):
        """Iterates over all unique tools in the collection."""
        return iter(self.tool_map.values())

    def to_params(self) -> List[Dict[str, Any]]:
        """
        Returns a list of OpenAI-compatible tool parameters for all unique tools
        in the collection.
        """
        return [tool.to_param() for tool in self.tool_map.values()]

    async def execute(
        self, *, name: str, tool_input: Dict[str, Any] = None
    ) -> ToolResult:
        """
        Executes a tool by name.

        Args:
            name: The name of the tool to execute.
            tool_input: A dictionary of inputs for the tool.

        Returns:
            A ToolResult or ToolFailure.
        """
        tool = self.get_tool(name) # Use get_tool to potentially access registry
        if not tool:
            return ToolFailure(error=f"Tool {name} is invalid or not found")
        try:
            result = await tool(**tool_input)
            return result
        except ToolError as e:
            return ToolFailure(error=e.message)

    async def execute_all(self) -> List[ToolResult]:
        """
        Execute all tools currently in this collection (local and from registry) sequentially.
        Note: This executes tools based on the current state of tool_map.
        """
        results = []
        for tool_name in self.tool_map: # Iterate using the consolidated tool_map
            tool = self.tool_map[tool_name]
            try:
                # Assuming tools can be called without specific input for execute_all
                # If tools require specific inputs, this method needs adjustment
                # or should only call tools that don't require inputs.
                # For now, let's assume they might have default behaviors or no-arg calls.
                if hasattr(tool, '__call__'):
                    result = await tool() 
                    results.append(result)
                else:
                    # This case should ideally not happen if BaseTool defines __call__
                    # or if execute_all is meant for specific tool types.
                    results.append(ToolFailure(error=f"Tool {tool.name} is not callable in execute_all"))
            except ToolError as e:
                results.append(ToolFailure(error=e.message))
            except Exception as e: # Catch other unexpected errors during tool execution
                logger.error(f"Unexpected error executing tool {tool.name}: {e}")
                results.append(ToolFailure(error=f"Unexpected error in tool {tool.name}: {str(e)}"))

        return results

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Retrieves a tool by its name from the collection or the linked registry.
        Explicitly added/managed tools in this collection take precedence.
        """
        tool = self.tool_map.get(name)
        if not tool and self.tool_registry:
            tool = self.tool_registry.get_tool(name)
            if tool: # If found in registry, add to local tool_map for faster access next time
                self.tool_map[name] = tool
        return tool

    def add_tool(self, tool: BaseTool):
        """
        Adds a single tool to this ToolCollection instance locally.
        This does NOT register the tool with the global ToolRegistry.
        If a tool with the same name already exists in this collection's local
        tool_map, it will be overwritten.
        """
        if not isinstance(tool, BaseTool):
            logger.warning(f"Attempted to add an object that is not a BaseTool: {getattr(tool, 'name', 'Unknown')}")
            return self

        self.tool_map[tool.name] = tool
        # Update self.tools to reflect the new state of tool_map
        self.tools = tuple(self.tool_map.values())
        
        # Also update _explicit_tools if we want add_tool to behave like an explicit addition
        # This ensures that if a tool with the same name was from the registry,
        # it's now considered "explicitly managed" by this collection.
        # Remove any existing tool with the same name from _explicit_tools first
        self._explicit_tools = [t for t in self._explicit_tools if t.name != tool.name]
        self._explicit_tools.append(tool)
        return self

    def add_tools(self, *tools: BaseTool):
        """
        Adds multiple tools to this ToolCollection instance locally.
        This does NOT register the tools with the global ToolRegistry.
        If any tool has a name conflict with an existing tool in this collection's
        local tool_map, it will be overwritten.
        """
        for tool in tools:
            self.add_tool(tool) # Leverages the logic in add_tool
        return self
