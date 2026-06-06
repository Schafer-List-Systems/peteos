#!/usr/bin/env python3
"""Data Cleansing example from the OAP API documentation.

Normalize messy, heterogeneous, or inconsistently formatted data records
using agent reasoning.

Usage:
    PYTHONPATH=/home/frygge/projects/private/peteos python examples/oap/01_data_cleansing.py \
        http://localhost:8080
"""

import sys
from dataclasses import dataclass

from peteos import AgenticObjectBase, Error, tool
from peteos.chatbot.manager import ChatBotManager


class PersonRecord(AgenticObjectBase):
    """A person record with messy personal data for normalization."""

    def __init__(self):
        super().__init__()
        self._full_name = "Dr. John Michael Smith Jr."
        self._height = "5'11\""
        self._birth_date = "03/15/1990"

    @tool
    def get_full_name(self) -> str:
        """Get the full name."""
        return self._full_name

    @tool
    def set_full_name(self, name: str) -> None:
        """Replace the full name."""
        self._full_name = name

    @tool
    def get_height(self) -> str:
        """Get the raw height string."""
        return self._height

    @tool
    def set_height(self, height: str) -> None:
        """Replace the height."""
        self._height = height

    @tool
    def get_birth_date(self) -> str:
        """Get the raw birth date string."""
        return self._birth_date

    @tool
    def set_birth_date(self, date: str) -> None:
        """Replace the birth date."""
        self._birth_date = date

    @tool
    def split_name(self, name: str) -> tuple[str, str, str, str]:
        """Split a name into title, first, middle, suffix."""
        parts = name.strip().split()
        title = parts[0] if parts and parts[0].endswith(".") else ""
        first = parts[1] if len(parts) > 1 else ""
        middle = parts[2] if len(parts) > 2 else ""
        suffix = (
            parts[-1]
            if len(parts) > 3 and parts[-1].endswith((".", "Jr.", "Sr."))
            else ""
        )
        return title, first, middle, suffix

    @tool
    def normalize_height(self, value: str) -> float:
        """Convert height to centimeters."""
        if "'" in value and '"' in value:
            inches = float(value.replace("'", "").replace('"', ""))
            return round(inches * 2.54, 2)
        if value.endswith("cm"):
            return float(value[:-2])
        if value.endswith("m"):
            return round(float(value[:-1]) * 100, 2)
        return float(value) * 2.54  # assume feet

    @tool
    def format_date_iso(self, date: str) -> str:
        """Convert various date formats to ISO 8601."""
        parts = date.split("/")
        if len(parts) == 3:
            mm, dd, yyyy = parts
            return f"{yyyy}-{mm.zfill(2)}-{dd.zfill(2)}"
        return date


@dataclass
class CleansingResult:
    name_parts: tuple[str, str, str, str]
    height_cm: float
    birth_iso: str


async def main():
    """Set up an Agent and invoke it on a PersonRecord."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <backend-url>")
        sys.exit(1)
    backend_url = sys.argv[1]

    # --- Set up peteos components ---
    chatbot_manager = ChatBotManager(timeout=60)
    await chatbot_manager.add_backend("local", backend_url)

    person = PersonRecord()

    # --- Invoke the agent ---
    try:
        result = await person.invoke_agent(
            prompt="Normalize this record: split the name, convert height to cm, "
            "and format the date as ISO 8601.",
            output_schema=CleansingResult,
            persistent_thread_id="clean-001",
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
