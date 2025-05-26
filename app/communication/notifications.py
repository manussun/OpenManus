from datetime import datetime
from pydantic import BaseModel, Field
from typing import List

class AttentionFlag(BaseModel):
    """
    Represents a flag raised by an agent for human attention.
    """
    agent_id: str = Field(..., description="The ID or name of the agent raising the flag.")
    message: str = Field(..., description="A message describing the reason for the flag.")
    urgency: int = Field(default=3, description="Urgency level (e.g., 1-low, 5-high).")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of when the flag was raised.")
    # Optional: Add context, suggested_actions, etc.

    def __str__(self) -> str:
        return f"[{self.timestamp.isoformat()}] URGENCY {self.urgency} from {self.agent_id}: {self.message}"

if __name__ == '__main__': # For basic testing
    flag = AttentionFlag(agent_id="TestAgent", message="This is a test flag.", urgency=4)
    print(flag)
    flag2 = AttentionFlag(agent_id="AnotherAgent", message="Low urgency test.")
    print(flag2)
