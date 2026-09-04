#!/usr/bin/env python3
"""Data Cleansing example — agentic objects for normalizing messy records.

Shows how agentic objects can reason about heterogeneous, inconsistently
formatted data and produce structured, normalized output using
invoke_agent with output_schema.
"""

from __future__ import annotations

from dataclasses import dataclass

from peteos import AgenticObject, Error, tool


@dataclass
class NormalizedPerson:
    """A normalized person record with structured, clean values."""

    title: str
    first_name: str
    middle_name: str
    last_name: str
    suffix: str
    height_cm: float
    birth_date: str  # ISO 8601 YYYY-MM-DD


class RecordNormalizer(AgenticObject):
    """You are a data normalizer for personal records.

    Parse the raw unstructured person record and produce a clean,
    structured NormalizedPerson.

    OUTPUT REQUIREMENTS:
    - Height MUST be in centimeters (float)
    - Date MUST be ISO 8601 (YYYY-MM-DD)
    - Use the provided conversion tools for computation. Do not guess
      arithmetic — call the tools.
    """

    def __init__(self, raw_record: str) -> None:
        super().__init__()
        self._raw_record = raw_record

    @tool(description="Return the raw person record string as received.")
    def get_raw_record(self) -> str:
        return self._raw_record

    @tool(description="Convert inches to centimeters. Multiply by 2.54.")
    def inches_to_cm(self, inches: float) -> float:
        return round(inches * 2.54, 2)

    @tool(description="Convert feet to centimeters. Multiply by 30.48.")
    def feet_to_cm(self, feet: float) -> float:
        return round(feet * 30.48, 2)

    @tool(description="Convert meters to centimeters. Multiply by 100.")
    def meters_to_cm(self, meters: float) -> float:
        return round(meters * 100, 2)

    @tool(description="Convert a date string to ISO 8601 format (YYYY-MM-DD).")
    def normalize_date(self, date: str) -> str:
        from datetime import datetime

        formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d %Y", "%b %d %Y"]
        for fmt in formats:
            try:
                return datetime.strptime(date.strip(), fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return date


async def normalize_person(raw_record: str) -> NormalizedPerson:
    """Normalize a person record using an agentic object.

    The agent reasons about the unstructured input, uses conversion
    tools for computation, and produces a structured NormalizedPerson
    with clean, standardized fields.
    """
    normalizer = RecordNormalizer(raw_record=raw_record)

    result = await normalizer.invoke_agent(
        prompt=f"Normalize this person record:\n\n{raw_record}",
        output_schema=NormalizedPerson,
        persistent_thread_id=None,
    )

    if isinstance(result, Error):
        raise ValueError(result.message)

    return result


async def main():
    """Run the normalizer on sample records."""

    print("=== Record 1: Dr. John Michael Smith Jr. ===")
    record = await normalize_person(
        raw_record="Name: Dr. John Michael Smith Jr., Height: 5'11\", DOB: 03/15/1990",
    )
    print(f"  {record}")

    print("\n=== Record 2: Maria Garcia ===")
    record2 = await normalize_person(
        raw_record="Name: Maria Garcia, Height: 165cm, DOB: 1988-07-22",
    )
    print(f"  {record2}")

    print("\n=== Record 3: Prof. Alice M. O'Brien III ===")
    record3 = await normalize_person(
        raw_record="Name: Prof. Alice M. O'Brien III, Height: 1.72m, DOB: December 3 1975",
    )
    print(f"  {record3}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
