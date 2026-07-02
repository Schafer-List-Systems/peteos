# Simulation Decision-Making

Use agent reasoning to drive decisions in simulations — game NPCs, robots, or any state-driven system.

## Example

```python
from dataclasses import dataclass
from peteos import AgenticObject, tool, invoke_agent

class LocationRecord(AgenticObject):
    def __init__(self, x: float = 0.0, y: float = 0.0):
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
        ...

class TaskRecord(AgenticObject):
    def __init__(self):
        self._description = "Fetch water"
        self._priority = 5
        self._target_location = LocationRecord()

    @tool
    def get_description(self) -> str:
        return self._description

    @tool
    def get_priority(self) -> int:
        return self._priority

    @tool
    def get_target_location(self) -> LocationRecord:
        return self._target_location

class NPC(AgenticObject):
    def __init__(self):
        self._position = LocationRecord()
        self._hunger = 75.0  # 0-100
        self._tasks = [TaskRecord()]

    @tool
    def get_position(self) -> LocationRecord:
        return self._position

    @tool
    def get_hunger(self) -> float:
        return self._hunger

    @tool
    def get_tasks(self) -> list[TaskRecord]:
        return self._tasks

    @tool
    def move_to(self, x: float, y: float) -> bool:
        """Move the NPC to a new position."""
        ...

@dataclass
class Decision:
    action: str
    target_x: float
    target_y: float
    reason: str

result = invoke_agent(
    NPC(),
    prompt="Decide what to do next. Prioritize by hunger level and task urgency.",
    output_schema=Decision,
    thread_id="npc-tick",
)
```

**Expected output:**

```python
Decision(
    action="move_to",
    target_x=45.2,
    target_y=12.8,
    reason="Hunger at 75% is approaching critical. Task 'Fetch water' has highest urgency. Target location is 30m away — move there now.",
)
```

## Why an agent?

| | Static Decision Tree | Agent-Based Decision |
|---|---|---|
| State at decision time | Known, fixed | Unknown, dynamic |
| Decision logic | Hand-written rules | Agent reasoning |
| Complexity handling | Scales poorly | Adapts to complexity |

The agentic object provides the structured state interface. The agent provides the reasoning engine. The simulation provides the execution layer.
