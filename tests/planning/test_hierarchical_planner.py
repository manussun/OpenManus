import unittest
import asyncio
from app.planning.planner import HierarchicalPlanner
from app.planning.structures import Plan, SubPlan

class TestHierarchicalPlanner(unittest.TestCase):
    """
    Unit tests for the HierarchicalPlanner class.
    Tests focus on the placeholder logic of decompose_task.
    """

    def setUp(self):
        """Set up for each test."""
        # Planner is initialized without LLM config for placeholder tests
        self.planner = HierarchicalPlanner()

    async def _run_decompose_task(self, task_description: str, agent_capabilities: list = None):
        """Helper to run async decompose_task."""
        if agent_capabilities is None:
            agent_capabilities = ["mock_capability_1", "mock_capability_2"]
        return await self.planner.decompose_task(task_description, agent_capabilities)

    def test_decompose_task_example_plan(self):
        """Test decompose_task with 'example plan' keyword."""
        task_desc = "Create an example plan for a demo."
        plan = asyncio.run(self._run_decompose_task(task_desc))

        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, task_desc)
        self.assertEqual(len(plan.sub_plans), 3)
        self.assertEqual(plan.status, 'pending')
        
        self.assertEqual(plan.sub_plans[0].task_id, "task_1")
        self.assertEqual(plan.sub_plans[0].description, "First example step: Do A")
        self.assertEqual(plan.sub_plans[1].task_id, "task_2")
        self.assertEqual(plan.sub_plans[1].description, "Second example step: Do B")
        self.assertEqual(plan.sub_plans[2].task_id, "task_3")
        self.assertEqual(plan.sub_plans[2].description, "Third example step: Verify C")
        for sp in plan.sub_plans:
            self.assertEqual(sp.status, 'pending')

    def test_decompose_task_write_code(self):
        """Test decompose_task with 'write code' keyword."""
        task_desc = "Write code for a new feature."
        plan = asyncio.run(self._run_decompose_task(task_desc))

        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, task_desc)
        self.assertEqual(len(plan.sub_plans), 4)
        self.assertEqual(plan.status, 'pending')

        expected_descriptions = [
            "Understand requirements for the code.",
            "Write the initial draft of the code.",
            "Test the written code.",
            "Refactor and finalize the code."
        ]
        expected_task_ids = ["code_1", "code_2", "code_3", "code_4"]

        for i, sp in enumerate(plan.sub_plans):
            self.assertEqual(sp.task_id, expected_task_ids[i])
            self.assertEqual(sp.description, expected_descriptions[i])
            self.assertEqual(sp.status, 'pending')

    def test_decompose_task_generic(self):
        """Test decompose_task with a generic task description."""
        task_desc = "Organize a birthday party."
        plan = asyncio.run(self._run_decompose_task(task_desc))

        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, task_desc)
        self.assertEqual(len(plan.sub_plans), 3)
        self.assertEqual(plan.status, 'pending')
        
        self.assertEqual(plan.sub_plans[0].task_id, "gen_1")
        self.assertEqual(plan.sub_plans[0].description, f"Understand the goal: {task_desc}")
        self.assertEqual(plan.sub_plans[1].task_id, "gen_2")
        self.assertEqual(plan.sub_plans[1].description, "Execute the core action for the goal.")
        self.assertEqual(plan.sub_plans[2].task_id, "gen_3")
        self.assertEqual(plan.sub_plans[2].description, "Verify the outcome of the action.")
        for sp in plan.sub_plans:
            self.assertEqual(sp.status, 'pending')

    def test_decompose_task_with_empty_capabilities(self):
        """Test decompose_task with empty agent_capabilities."""
        task_desc = "Generic task with empty capabilities."
        plan = asyncio.run(self._run_decompose_task(task_desc, agent_capabilities=[]))
        
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, task_desc)
        # Should still produce the generic plan as capabilities are not used by placeholder
        self.assertEqual(len(plan.sub_plans), 3) 
        self.assertEqual(plan.sub_plans[0].task_id, "gen_1")

    def test_decompose_task_with_none_capabilities(self):
        """Test decompose_task with None agent_capabilities."""
        task_desc = "Generic task with None capabilities."
        plan = asyncio.run(self.planner.decompose_task(task_desc, None)) # Directly pass None
        
        self.assertIsInstance(plan, Plan)
        self.assertEqual(plan.goal, task_desc)
        self.assertEqual(len(plan.sub_plans), 3)
        self.assertEqual(plan.sub_plans[0].task_id, "gen_1")

if __name__ == '__main__':
    unittest.main()
