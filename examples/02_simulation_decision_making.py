#!/usr/bin/env python3
"""Simulation Decision-Making example from the OAP API documentation.

Use agent reasoning to drive decisions in simulations — game NPCs, robots,
or any state-driven system.
"""

from dataclasses import dataclass
from peteos import AgenticObject, Error, tool


class LocationRecord(AgenticObject):
    """A position in a 2D simulation space."""

    def __init__(self, x: float = 0.0, y: float = 0.0):
        super().__init__()
        self._x = x
        self._y = y

    @tool
    def get_x(self) -> float:
        """Get x coordinate."""
        return self._x

    @tool
    def get_y(self) -> float:
        """Get y coordinate."""
        return self._y

    @tool
    def distance_to(self, other: "LocationRecord") -> float:
        """Calculate Euclidean distance to another location."""
        import math

        return math.sqrt((self._x - other._x) ** 2 + (self._y - other._y) ** 2)

    @tool
    def set_position(self, x: float, y: float) -> None:
        """Set position coordinates."""
        self._x = x
        self._y = y


class TaskRecord(AgenticObject):
    """A task in the simulation."""

    def __init__(self):
        super().__init__()
        self._description = "Fetch water"
        self._priority = 5
        self._target_location = LocationRecord(x=45.0, y=12.0)

    @tool
    def get_description(self) -> str:
        """Get task description."""
        return self._description

    @tool
    def get_priority(self) -> int:
        """Get task priority."""
        return self._priority

    @tool
    def get_target_location(self) -> LocationRecord:
        """Get task target location."""
        return self._target_location

    @tool
    def complete_task(self) -> str:
        """Mark the task as complete."""
        return f"Completed: {self._description}"


@dataclass
class Decision:
    action: str
    target_x: float
    target_y: float
    reason: str


class NPC(AgenticObject):
    """A simulated character that reasons over its state to decide actions."""

    def __init__(self):
        super().__init__()
        self._position = LocationRecord(x=15.2, y=8.5)
        self._hunger = 75.0  # 0-100 scale
        self._tasks = [TaskRecord()]

    @tool
    def get_position(self) -> LocationRecord:
        """Get the NPC's current position."""
        return self._position

    @tool
    def get_hunger(self) -> float:
        """Get the NPC's hunger level (0-100)."""
        return self._hunger

    @tool
    def get_tasks(self) -> list[TaskRecord]:
        """Get the NPC's current tasks."""
        return self._tasks

    @tool
    def move_to(self, x: float, y: float) -> bool:
        """Move the NPC to a new position."""
        self._position.set_position(x, y)
        return True

    @tool
    def rest(self) -> str:
        """Rest to recover energy."""
        self._hunger = max(0.0, self._hunger - 10.0)
        return f"Rested. Hunger now: {self._hunger:.1f}"


async def main():
    """Set up an Agent and invoke it on an NPC for decision making."""
    npc = NPC()

    # --- Invoke the agent ---
    try:
        result = await npc.invoke_agent(
            prompt="Decide what to do next. Prioritize by hunger level and "
            "task urgency. Return structured output with your decision.",
            output_schema=Decision,
            persistent_thread_id="npc-tick",
        )
        if isinstance(result, Error):
            print(f"Error: {result.message}")
        else:
            print(f"Result: {result}")
    except Exception as e:
        print(f"API failure: {e!r}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
