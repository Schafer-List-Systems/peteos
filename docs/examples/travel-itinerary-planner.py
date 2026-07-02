"""Example: Travel Itinerary Planner — agentic object with tools, sandbox code, and structured output.

Usage: python travel-itinerary-planner.py <backend-url>
"""

import asyncio
import sys
from dataclasses import dataclass
from enum import Enum

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObject
from peteos.oap.decorators import agentic_object, tool


class ActivityType(Enum):
    SIGHTSEEING = "Sightseeing"
    FOOD = "Food"
    TRANSPORT = "Transport"
    ACCOMMODATION = "Accommodation"
    CULTURE = "Culture"


@agentic_object(allow_code_execution=True)
class TravelItineraryPlanner(AgenticObject):
    """You are a travel itinerary planner. You manage destination details,
    budget, and travel days. Use set_destination, set_budget, and set_days
    to configure the trip. Add activities with add_activity and view them
    with list_activities. The agent can plan a structured itinerary using
    recommend_itinerary."""

    def __init__(self):
        super().__init__()
        self._destination: str = ""
        self._budget: float = 0.0
        self._days: int = 0
        self._activities: list[dict] = []

    @tool
    def set_destination(self, destination: str) -> str:
        """Set the travel destination city."""
        self._destination = destination
        return f"Destination set to {destination}."

    @tool
    def set_budget(self, budget: float) -> str:
        """Set the total travel budget."""
        self._budget = budget
        return f"Budget set to {budget:.2f}."

    @tool
    def set_days(self, days: int) -> str:
        """Set the number of travel days."""
        self._days = days
        return f"Trip duration set to {days} day(s)."

    @tool
    def add_activity(self, day: int, activity_type: ActivityType, name: str, cost: float) -> str:
        """Add an activity for a specific day with type, name, and estimated cost."""
        self._activities.append({
            "day": day,
            "type": activity_type.value,
            "name": name,
            "cost": cost,
        })
        return f"Added {name} on day {day} — {cost:.2f}."

    @tool
    def list_activities(self) -> list[dict]:
        """Return all planned activities sorted by day."""
        return sorted(self._activities, key=lambda a: a["day"])


@dataclass
class DailyPlan:
    day: int
    activities: list[str]
    daily_cost: float


@dataclass
class ItineraryRecommendation:
    destination: str
    total_days: int
    daily_plans: list[DailyPlan]
    remaining_budget: float


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    args = sys.argv[1:] + [None] * 3
    url, api_type, api_key = args[:3]
    await ChatBotManager.add_backend("local", url, api_type=api_type, api_key=api_key)

    planner = TravelItineraryPlanner()

    # Configure trip
    await planner.invoke_agent("Set destination to Tokyo, budget to 3000, and 3 days.")
    await planner.invoke_agent(
        "Add activities: day 1 — Tokyo Tower sightseeing 50, "
        "sushi dinner food 40, hotel accommodation 100; "
        "day 2 — Shibuya culture 30, ramen food 15; "
        "day 3 — airport transport 20."
    )

    # Get structured itinerary recommendation
    result = await planner.invoke_agent(
        "Recommend a day-by-day itinerary based on the planned activities.",
        output_schema=ItineraryRecommendation,
    )
    print("Itinerary recommendation:", result)

    # Use sandbox to optimize remaining budget
    result = await planner.invoke_agent(
        "How much budget remains after all planned activities? Suggest one more activity with remaining funds.",
        output_schema=ItineraryRecommendation,
    )
    print("Optimized itinerary:", result)


if __name__ == "__main__":
    asyncio.run(main())
