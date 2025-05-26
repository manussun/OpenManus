from typing import List, Optional
from pydantic import BaseModel, Field

class SubPlan(BaseModel):
    """
    Represents a single task or step within a larger plan.
    """
    task_id: str = Field(..., description="Unique identifier for the sub-task.")
    description: str = Field(..., description="Detailed description of the sub-task.")
    status: str = Field(default='pending', description="Current status of the sub-task (e.g., 'pending', 'in_progress', 'completed', 'failed').")
    result: Optional[str] = Field(default=None, description="Outcome or result of the sub-task upon completion or failure.")
    # Optional: Add parent_task_id if sub-plans can be nested further, or dependencies List[str] for DAGs

class Plan(BaseModel):
    """
    Represents a high-level plan composed of multiple sub-plans (tasks).
    """
    goal: str = Field(..., description="The overall goal that this plan aims to achieve.")
    sub_plans: List[SubPlan] = Field(default_factory=list, description="A list of sub-plans (tasks) to achieve the goal.")
    status: str = Field(default='pending', description="Current status of the overall plan (e.g., 'pending', 'in_progress', 'completed', 'failed').")
    current_sub_plan_index: int = Field(default=0, description="Index of the currently active or next sub-plan.")

    def get_current_sub_plan(self) -> Optional[SubPlan]:
        """Returns the current sub-plan based on the index, or None if out of bounds."""
        if 0 <= self.current_sub_plan_index < len(self.sub_plans):
            return self.sub_plans[self.current_sub_plan_index]
        return None

    def advance_to_next_sub_plan(self) -> Optional[SubPlan]:
        """Moves to the next sub-plan and returns it. Returns None if no more sub-plans."""
        self.current_sub_plan_index += 1
        return self.get_current_sub_plan()

    def update_sub_plan_status(self, task_id: str, status: str, result: Optional[str] = None) -> bool:
        """Updates the status and result of a specific sub-plan."""
        for sub_plan in self.sub_plans:
            if sub_plan.task_id == task_id:
                sub_plan.status = status
                sub_plan.result = result
                # Optionally, update overall plan status based on sub-plan statuses
                self._update_overall_status()
                return True
        return False # Task ID not found

    def _update_overall_status(self) -> None:
        """Updates the overall plan status based on the statuses of its sub-plans."""
        if all(sp.status == 'completed' for sp in self.sub_plans):
            self.status = 'completed'
        elif any(sp.status == 'failed' for sp in self.sub_plans):
            self.status = 'failed'
        elif any(sp.status == 'in_progress' for sp in self.sub_plans) or \
             (any(sp.status == 'pending' for sp in self.sub_plans) and self.status == 'in_progress'): # if some started, it's in_progress
            self.status = 'in_progress'
        elif all(sp.status == 'pending' for sp in self.sub_plans) and self.current_sub_plan_index == 0 :
             self.status = 'pending' # Explicitly set to pending if no sub_plan has started
        # Add more sophisticated logic as needed, e.g., partial completion, etc.
        
        # If all sub_plans are completed, set the overall plan to completed.
        if self.status != 'failed' and self.current_sub_plan_index >= len(self.sub_plans) and \
           all(sp.status == 'completed' for sp in self.sub_plans):
            self.status = 'completed'


    def is_completed(self) -> bool:
        """Checks if all sub-plans are completed."""
        return all(sp.status == 'completed' for sp in self.sub_plans)

    def has_failed(self) -> bool:
        """Checks if any sub-plan has failed."""
        return any(sp.status == 'failed' for sp in self.sub_plans)
