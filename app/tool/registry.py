import threading
from typing import Dict, List, Optional
from app.tool.base import BaseTool # Assuming BaseTool is in app.tool.base

class ToolRegistry:
    """
    A thread-safe registry for managing tools available to agents.
    """
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._lock = threading.Lock()

    def register_tool(self, tool: BaseTool, overwrite: bool = False) -> None:
        """
        Registers a tool.

        Args:
            tool: The tool instance to register.
            overwrite: If True, overwrite an existing tool with the same name. 
                       Otherwise, a warning is logged and the tool is not added.
        """
        if not isinstance(tool, BaseTool):
            # Log or raise an error: tool is not an instance of BaseTool
            print(f"Warning: Attempted to register an object that is not a BaseTool: {tool.name if hasattr(tool, 'name') else 'Unknown'}")
            return

        with self._lock:
            if tool.name in self._tools and not overwrite:
                # Log a warning about the name conflict
                print(f"Warning: Tool with name '{tool.name}' already registered. Skipping.")
                return
            self._tools[tool.name] = tool
            print(f"Tool '{tool.name}' registered.")

    def unregister_tool(self, tool_name: str) -> None:
        """
        Removes a tool from the registry.

        Args:
            tool_name: The name of the tool to unregister.
        """
        with self._lock:
            if tool_name in self._tools:
                del self._tools[tool_name]
                print(f"Tool '{tool_name}' unregistered.")
            else:
                # Log a warning that the tool was not found
                print(f"Warning: Tool with name '{tool_name}' not found for unregistration.")

    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """
        Retrieves a tool by its name.

        Args:
            tool_name: The name of the tool to retrieve.

        Returns:
            The tool instance if found, otherwise None.
        """
        with self._lock:
            return self._tools.get(tool_name)

    def list_tools(self) -> List[BaseTool]:
        """
        Returns a list of all registered tools.

        Returns:
            A list of BaseTool instances.
        """
        with self._lock:
            return list(self._tools.values())

    def get_all_tool_names(self) -> List[str]:
        """
        Returns a list of names of all registered tools.

        Returns:
            A list of tool names.
        """
        with self._lock:
            return list(self._tools.keys())
