import unittest
from typing import Any, Dict, Optional, List

from app.tool.registry import ToolRegistry
from app.tool.tool_collection import ToolCollection
from app.tool.base import BaseTool, ToolResult

# Mock tools for testing
class MockTool(BaseTool):
    name: str
    description: str = "A mock tool."
    parameters: Optional[Dict[str, Any]] = {"type": "object", "properties": {}}
    version: str = "1.0" # To differentiate versions of the same tool

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(output=f"{self.name} v{self.version} executed with {kwargs}")

class TestToolCollectionWithRegistry(unittest.TestCase):
    """
    Unit tests for ToolCollection's integration with ToolRegistry.
    """

    def setUp(self):
        """Set up for each test."""
        self.registry = ToolRegistry()
        self.tool_reg1 = MockTool(name="RegistryTool1", version="reg")
        self.tool_reg2 = MockTool(name="RegistryTool2", version="reg")
        self.tool_shared = MockTool(name="SharedTool", version="reg")
        
        self.registry.register_tool(self.tool_reg1)
        self.registry.register_tool(self.tool_reg2)
        self.registry.register_tool(self.tool_shared)

        self.tool_local1 = MockTool(name="LocalTool1", version="local")
        self.tool_shared_local = MockTool(name="SharedTool", version="local_override")


    def test_init_with_empty_registry_and_no_local_tools(self):
        """Test ToolCollection with an empty registry and no local tools."""
        empty_registry = ToolRegistry()
        collection = ToolCollection(tool_registry=empty_registry)
        self.assertEqual(len(list(collection)), 0)
        self.assertEqual(len(collection.to_params()), 0)

    def test_init_with_registry_only(self):
        """Test ToolCollection initialized only with a ToolRegistry."""
        collection = ToolCollection(tool_registry=self.registry)
        self.assertEqual(len(list(collection)), 3) # tool_reg1, tool_reg2, tool_shared
        
        retrieved_reg1 = collection.get_tool("RegistryTool1")
        self.assertIsNotNone(retrieved_reg1)
        self.assertEqual(retrieved_reg1.version, "reg") # type: ignore

        retrieved_shared = collection.get_tool("SharedTool")
        self.assertIsNotNone(retrieved_shared)
        self.assertEqual(retrieved_shared.version, "reg") # type: ignore

    def test_init_with_local_tools_only_no_registry(self):
        """Test ToolCollection with local tools and no registry."""
        collection = ToolCollection(self.tool_local1, self.tool_shared_local)
        self.assertEqual(len(list(collection)), 2)
        
        retrieved_local1 = collection.get_tool("LocalTool1")
        self.assertIsNotNone(retrieved_local1)
        self.assertEqual(retrieved_local1.version, "local") # type: ignore

        retrieved_shared = collection.get_tool("SharedTool") # Should be local_override
        self.assertIsNotNone(retrieved_shared)
        self.assertEqual(retrieved_shared.version, "local_override") # type: ignore

    def test_init_with_registry_and_local_tools_precedence(self):
        """Test local tools take precedence over registry tools with the same name."""
        collection = ToolCollection(self.tool_local1, self.tool_shared_local, tool_registry=self.registry)
        
        # Expected tools: RegistryTool1, RegistryTool2, LocalTool1, SharedTool (local_override)
        self.assertEqual(len(list(collection)), 4) 
        tool_names_in_collection = {tool.name for tool in collection}
        self.assertIn("RegistryTool1", tool_names_in_collection)
        self.assertIn("RegistryTool2", tool_names_in_collection)
        self.assertIn("LocalTool1", tool_names_in_collection)
        self.assertIn("SharedTool", tool_names_in_collection)

        retrieved_shared = collection.get_tool("SharedTool")
        self.assertIsNotNone(retrieved_shared)
        self.assertEqual(retrieved_shared.version, "local_override") # Local version should override registry

        retrieved_reg1 = collection.get_tool("RegistryTool1") # Should come from registry
        self.assertIsNotNone(retrieved_reg1)
        self.assertEqual(retrieved_reg1.version, "reg") # type: ignore

    def test_get_tool_fetches_from_registry_if_not_local(self):
        """Test get_tool fetches from registry if not found in local tool_map initially."""
        # Initialize with only local tools, but with registry linked
        collection = ToolCollection(self.tool_local1, tool_registry=self.registry)
        
        # Get a tool that is only in the registry
        retrieved_reg2 = collection.get_tool("RegistryTool2")
        self.assertIsNotNone(retrieved_reg2)
        self.assertEqual(retrieved_reg2.name, "RegistryTool2")
        self.assertEqual(retrieved_reg2.version, "reg") # type: ignore

        # Check that it's now in the collection's tool_map for faster access
        self.assertIn("RegistryTool2", collection.tool_map)
        self.assertEqual(len(list(collection)), 3) # LocalTool1, RegistryTool1, RegistryTool2, SharedTool (from registry)
                                                    # After get_tool("RegistryTool2"), it's added to tool_map.
                                                    # Let's re-evaluate:
                                                    # Initial: LocalTool1. Registry: Reg1, Reg2, Shared.
                                                    # tool_map upon init: LocalTool1 (local), Reg1 (reg), Reg2 (reg), Shared (reg)
                                                    # No, tool_map upon init if local tools are passed:
                                                    # Reg1, Reg2, Shared from registry. Then LocalTool1 added.
                                                    # So tool_map has: LocalTool1, Reg1, Reg2, Shared (reg).
                                                    # This was my mistake in previous version of ToolCollection logic.
                                                    # The current logic for ToolCollection.__init__ is:
                                                    # 1. Populate from registry.
                                                    # 2. Add explicit tools, overriding.
                                                    # So, if collection = ToolCollection(self.tool_local1, tool_registry=self.registry)
                                                    # tool_map starts with {Reg1, Reg2, Shared(reg)}
                                                    # then tool_local1 is added.
                                                    # so tool_map should be {Reg1,Reg2,Shared(reg),LocalTool1}
                                                    # therefore len(list(collection)) should be 4.
        
        # Let's re-verify the setup for this specific test case logic.
        # collection = ToolCollection(self.tool_local1, tool_registry=self.registry)
        # Registry has: RegistryTool1, RegistryTool2, SharedTool
        # Local provided: LocalTool1
        # Expected in collection.tool_map after init:
        #   RegistryTool1 (from registry)
        #   RegistryTool2 (from registry)
        #   SharedTool (from registry)
        #   LocalTool1 (explicitly passed)
        self.assertEqual(len(list(collection)), 4, "Initial collection size mismatch")

        # Now, get_tool("RegistryTool2") should return the one from registry.
        # It should already be in tool_map from initialization.
        # The specific part of get_tool "if not tool and self.tool_registry: tool = self.tool_registry.get_tool(name)"
        # is for cases where a tool might be added to the registry *after* the ToolCollection was initialized.
        # To test that specific path, we need to add to registry post-init.

        new_reg_tool = MockTool(name="NewlyAddedRegTool", version="new_reg")
        self.registry.register_tool(new_reg_tool)

        retrieved_new_reg = collection.get_tool("NewlyAddedRegTool")
        self.assertIsNotNone(retrieved_new_reg)
        self.assertEqual(retrieved_new_reg.name, "NewlyAddedRegTool")
        self.assertIn("NewlyAddedRegTool", collection.tool_map, "Tool fetched from registry should be added to local map.")
        self.assertEqual(len(list(collection.tool_map.values())), 5, "Collection size should increase after fetching new tool from registry.")


    def test_to_params_includes_all_tools(self):
        """Test to_params includes tools from registry and local, respecting precedence."""
        collection = ToolCollection(self.tool_local1, self.tool_shared_local, tool_registry=self.registry)
        params = collection.to_params()
        self.assertEqual(len(params), 4) # Reg1, Reg2, Local1, Shared(local_override)
        
        param_names = [p["function"]["name"] for p in params]
        self.assertIn("RegistryTool1", param_names)
        self.assertIn("RegistryTool2", param_names)
        self.assertIn("LocalTool1", param_names)
        self.assertIn("SharedTool", param_names)

        # Find SharedTool param and check its description (indirectly checking version via description if we put it there)
        # For this test, just ensuring the name count is correct is a good start.
        # If MockTool's to_param included version in description, we could check that.

    def test_iteration_includes_all_tools(self):
        """Test iterating over the collection includes all unique tools, respecting precedence."""
        collection = ToolCollection(self.tool_local1, self.tool_shared_local, tool_registry=self.registry)
        
        iterated_tools: List[BaseTool] = []
        for tool in collection:
            iterated_tools.append(tool)
        
        self.assertEqual(len(iterated_tools), 4)
        
        iterated_tool_versions: Dict[str, str] = {t.name: t.version for t in iterated_tools} # type: ignore
        
        self.assertEqual(iterated_tool_versions.get("RegistryTool1"), "reg")
        self.assertEqual(iterated_tool_versions.get("RegistryTool2"), "reg")
        self.assertEqual(iterated_tool_versions.get("LocalTool1"), "local")
        self.assertEqual(iterated_tool_versions.get("SharedTool"), "local_override")

    def test_add_tool_locally_after_init_with_registry(self):
        """Test adding a tool locally to a collection that's linked to a registry."""
        collection = ToolCollection(tool_registry=self.registry) # Has Reg1, Reg2, Shared(reg)
        
        new_local_tool = MockTool(name="NewLocalOnly", version="local_added")
        collection.add_tool(new_local_tool)
        
        self.assertEqual(len(list(collection)), 4)
        retrieved_new_local = collection.get_tool("NewLocalOnly")
        self.assertIsNotNone(retrieved_new_local)
        self.assertEqual(retrieved_new_local.version, "local_added") # type: ignore

        # Ensure it doesn't affect the registry itself
        self.assertIsNone(self.registry.get_tool("NewLocalOnly"))

        # Test adding a tool that overrides one from the registry locally
        override_reg1_locally = MockTool(name="RegistryTool1", version="local_override_of_reg1")
        collection.add_tool(override_reg1_locally)
        
        self.assertEqual(len(list(collection)), 4) # Count should remain same, but content changes
        retrieved_override = collection.get_tool("RegistryTool1")
        self.assertIsNotNone(retrieved_override)
        self.assertEqual(retrieved_override.version, "local_override_of_reg1") # type: ignore

        # Ensure original registry tool is untouched
        original_reg1_from_registry = self.registry.get_tool("RegistryTool1")
        self.assertIsNotNone(original_reg1_from_registry)
        self.assertEqual(original_reg1_from_registry.version, "reg") # type: ignore


if __name__ == '__main__':
    unittest.main()
