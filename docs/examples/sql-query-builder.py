"""Example: SQL Query Builder — agentic object with tools, sandbox code, and structured output.

Usage: python sql-query-builder.py <backend-url>
"""

import asyncio
import sys
from dataclasses import dataclass, field
from typing import Optional

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import agentic_object, tool


@dataclass
class Column:
    name: str
    data_type: str
    nullable: bool


@dataclass
class TableInfo:
    name: str
    columns: list[Column]


@agentic_object(allow_code_execution=True)
class SQLQueryBuilder(AgenticObjectBase):
    """You are a SQL query builder. You manage a database schema and help
    construct safe SQL queries. Use list_tables and describe_table to
    explore the schema. The agent can build and validate queries using
    Python sandbox execution and return structured results."""

    def __init__(self):
        super().__init__()
        self._schema: list[TableInfo] = [
            TableInfo("users", [
                Column("id", "INTEGER", False),
                Column("name", "TEXT", False),
                Column("email", "TEXT", True),
                Column("role", "TEXT", False),
            ]),
            TableInfo("orders", [
                Column("id", "INTEGER", False),
                Column("user_id", "INTEGER", False),
                Column("product", "TEXT", False),
                Column("amount", "FLOAT", False),
                Column("status", "TEXT", False),
            ]),
            TableInfo("products", [
                Column("id", "INTEGER", False),
                Column("name", "TEXT", False),
                Column("price", "FLOAT", False),
                Column("category", "TEXT", True),
            ]),
        ]

    @tool
    def list_tables(self) -> list[str]:
        """Return the names of all tables in the schema."""
        return [t.name for t in self._schema]

    @tool
    def describe_table(self, table_name: str) -> dict:
        """Describe columns of a table with names, data types, and nullability."""
        for t in self._schema:
            if t.name == table_name:
                return {
                    "table": t.name,
                    "columns": [
                        {"name": c.name, "type": c.data_type, "nullable": c.nullable}
                        for c in t.columns
                    ],
                }
        return {"error": f"Unknown table: {table_name}"}

    @tool
    def get_table(self, table_name: str) -> Optional[list[dict]]:
        """Return column info for a specific table as structured data."""
        for t in self._schema:
            if t.name == table_name:
                return [{"name": c.name, "type": c.data_type} for c in t.columns]
        return None


@dataclass
class QueryValidation:
    table: str
    operation: str
    columns_used: list[str]
    is_safe: bool
    warnings: list[str]


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    url = sys.argv[1]
    await ChatBotManager.add_backend("local", url)

    builder = SQLQueryBuilder()

    # Explore schema
    await builder.invoke_agent("List all tables and describe the orders table.")

    # Build and validate a query
    result = await builder.invoke_agent(
        "Build a SELECT query to get total amount per user from orders, "
        "only for users with at least 2 orders, and validate it is safe.",
        output_schema=QueryValidation,
    )
    print("Query validation:", result)

    # Verify column references exist in the schema
    result = await builder.invoke_agent(
        "Build a query selecting active products with their prices, then verify "
        "all column references exist in the products table.",
        output_schema=QueryValidation,
    )
    print("Column verification:", result)


if __name__ == "__main__":
    asyncio.run(main())
