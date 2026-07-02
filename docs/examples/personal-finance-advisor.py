"""Example: Personal Finance Advisor — agentic object with tools, sandbox code, and structured output.

Usage: python personal-finance-advisor.py <backend-url>
"""

import asyncio
import sys
from dataclasses import dataclass
from enum import Enum

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool


class Category(Enum):
    INCOME = "Income"
    FOOD = "Food"
    TRANSPORT = "Transport"
    HOUSING = "Housing"
    ENTERTAINMENT = "Entertainment"


@agentic_object(allow_code_execution=True)
class PersonalFinanceAdvisor(AgenticObjectBase):
    """You are a personal finance advisor. You manage income, track expenses,
    and provide financial recommendations. Use list_entries and add_entry
    to interact with the transaction ledger, and recommend_budget to get
    a structured budget recommendation."""

    def __init__(self):
        super().__init__()
        self._income: float = 0.0
        self._expenses: list[dict] = []

    @tool
    def set_income(self, amount: float) -> str:
        """Set the monthly income."""
        self._income = amount
        return f"Income set to {amount:.2f}."

    @tool
    def add_expense(self, category: Category, amount: float, description: str) -> str:
        """Add an expense entry with category, amount, and description."""
        self._expenses.append({"category": category.value, "amount": amount, "description": description})
        return f"Added expense: {description} ({category.value}) — {amount:.2f}."

    @tool
    def list_expenses(self) -> list[dict]:
        """Return all expense entries."""
        return list(self._expenses)

    @tool
    def get_income(self) -> float:
        """Return the monthly income."""
        return self._income


@dataclass
class BudgetRecommendation:
    category: str
    recommended_amount: float
    percentage_of_income: float


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    args = sys.argv[1:] + [None] * 3
    url, api_type, api_key = args[:3]
    await ChatBotManager.add_backend("local", url, api_type=api_type, api_key=api_key)

    advisor = PersonalFinanceAdvisor()

    # Set income and add expenses
    await advisor.invoke_agent("Set my monthly income to 5000.")
    await advisor.invoke_agent(
        "Add expenses: rent 1500 for housing, groceries 400 for food, "
        "bus pass 80 for transport, dinner out 60 for entertainment."
    )

    # Get structured budget recommendation
    result = await advisor.invoke_agent(
        "Give me a budget recommendation based on the current expenses.",
        output_schema=list[BudgetRecommendation],
    )
    print("Budget recommendation:", result)

    # Use sandbox to calculate savings rate
    result = await advisor.invoke_agent(
        "Calculate how much I can save each month and what percentage of my income that is.",
        output_schema=BudgetRecommendation,
    )
    print("Savings:", result)


if __name__ == "__main__":
    asyncio.run(main())
