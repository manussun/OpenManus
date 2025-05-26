import unittest
import threading
import time
from typing import Any, Dict, Optional
from app.tool.registry import ToolRegistry
from app.tool.base import BaseTool, ToolResult # Assuming BaseTool and ToolResult are in app.tool.base

# Dummy tool for testing
class MockTool(BaseTool):
    name: str
    description: str = "A mock tool for testing."
    parameters: Optional[Dict[str, Any]] = {"type": "object", "properties": {}}

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output=f"{self.name} executed with {kwargs}")

class TestToolRegistry(unittest.TestCase):
    """
    Unit tests for the ToolRegistry class.
    """

    def setUp(self):
        """Set up for each test."""
        self.registry = ToolRegistry()
        self.tool1 = MockTool(name="ToolA")
        self.tool2 = MockTool(name="ToolB")

    def test_register_and_get_tool(self):
        """Test registering a new tool and retrieving it."""
        self.registry.register_tool(self.tool1)
        retrieved_tool = self.registry.get_tool("ToolA")
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.name, "ToolA")

    def test_get_non_existent_tool(self):
        """Test retrieving a non-existent tool returns None."""
        retrieved_tool = self.registry.get_tool("NonExistentTool")
        self.assertIsNone(retrieved_tool)

    def test_list_tools(self):
        """Test listing all registered tools."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        tools = self.registry.list_tools()
        self.assertEqual(len(tools), 2)
        tool_names = [t.name for t in tools]
        self.assertIn("ToolA", tool_names)
        self.assertIn("ToolB", tool_names)

    def test_get_all_tool_names(self):
        """Test getting all tool names."""
        self.registry.register_tool(self.tool1)
        self.registry.register_tool(self.tool2)
        tool_names = self.registry.get_all_tool_names()
        self.assertEqual(len(tool_names), 2)
        self.assertIn("ToolA", tool_names)
        self.assertIn("ToolB", tool_names)


    def test_unregister_tool(self):
        """Test unregistering an existing tool."""
        self.registry.register_tool(self.tool1)
        self.assertIsNotNone(self.registry.get_tool("ToolA"))
        self.registry.unregister_tool("ToolA")
        self.assertIsNone(self.registry.get_tool("ToolA"))
        self.assertEqual(len(self.registry.list_tools()), 0)

    def test_unregister_non_existent_tool(self):
        """Test unregistering a non-existent tool does not raise an error."""
        try:
            self.registry.unregister_tool("NonExistentTool")
        except Exception as e:
            self.fail(f"Unregistering non-existent tool raised an exception: {e}")
        # Ensure no side-effects if other tools exist
        self.registry.register_tool(self.tool1)
        self.registry.unregister_tool("NonExistentTool")
        self.assertIsNotNone(self.registry.get_tool("ToolA"))


    def test_register_duplicate_tool_no_overwrite(self):
        """Test registering a tool with a duplicate name (default: no overwrite)."""
        self.registry.register_tool(self.tool1)
        tool_a_v2 = MockTool(name="ToolA", description="Version 2 of ToolA")
        
        # In a real scenario, we'd capture logs. Here, we check behavior.
        # Assuming the default behavior is to not overwrite and potentially log a warning.
        self.registry.register_tool(tool_a_v2) # overwrite=False by default
        
        retrieved_tool = self.registry.get_tool("ToolA")
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.description, "A mock tool for testing.") # Original description
        self.assertEqual(len(self.registry.list_tools()), 1)

    def test_register_duplicate_tool_with_overwrite(self):
        """Test registering a tool with a duplicate name and overwrite=True."""
        self.registry.register_tool(self.tool1)
        tool_a_v2 = MockTool(name="ToolA", description="Version 2 of ToolA")
        self.registry.register_tool(tool_a_v2, overwrite=True)
        
        retrieved_tool = self.registry.get_tool("ToolA")
        self.assertIsNotNone(retrieved_tool)
        self.assertEqual(retrieved_tool.description, "Version 2 of ToolA") # New description
        self.assertEqual(len(self.registry.list_tools()), 1)

    def test_register_non_basetool_instance(self):
        """Test attempting to register an object that is not a BaseTool instance."""
        class NotATool:
            name = "InvalidTool"
            description = "Not a BaseTool"

        invalid_tool = NotATool()
        # This should ideally log a warning and not add the tool.
        # We can't easily check logs here, so we check that the tool wasn't added.
        self.registry.register_tool(invalid_tool) # type: ignore 
        self.assertIsNone(self.registry.get_tool("InvalidTool"))
        self.assertEqual(len(self.registry.list_tools()), 0)


    def _thread_register_tools(self, num_tools, prefix):
        for i in range(num_tools):
            tool = MockTool(name=f"{prefix}_ThreadTool_{i}")
            self.registry.register_tool(tool)
            time.sleep(0.001) # Small sleep

    def _thread_get_tools(self, num_tools, prefix):
        for i in range(num_tools):
            self.registry.get_tool(f"{prefix}_ThreadTool_{i}")
            time.sleep(0.001)

    def test_thread_safety_register_get(self):
        """Test thread safety for registration and retrieval."""
        num_items_per_thread = 20
        
        thread1_register = threading.Thread(target=self._thread_register_tools, args=(num_items_per_thread, "T1"))
        thread2_register = threading.Thread(target=self._thread_register_tools, args=(num_items_per_thread, "T2"))
        thread3_get_t1 = threading.Thread(target=self._thread_get_tools, args=(num_items_per_thread, "T1"))
        thread4_get_t2 = threading.Thread(target=self._thread_get_tools, args=(num_items_per_thread, "T2"))

        thread1_register.start()
        thread2_register.start()
        # Give some time for registration threads to start
        time.sleep(0.01) 
        thread3_get_t1.start()
        thread4_get_t2.start()

        thread1_register.join()
        thread2_register.join()
        thread3_get_t1.join()
        thread4_get_t2.join()

        self.assertEqual(len(self.registry.list_tools()), num_items_per_thread * 2)
        for i in range(num_items_per_thread):
            self.assertIsNotNone(self.registry.get_tool(f"T1_ThreadTool_{i}"))
            self.assertIsNotNone(self.registry.get_tool(f"T2_ThreadTool_{i}"))
        print(f"\nToolRegistry thread safety test: completed {num_items_per_thread*2} registrations and gets.")


if __name__ == '__main__':
    unittest.main()
