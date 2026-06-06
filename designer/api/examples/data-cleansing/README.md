# Data Cleansing

Normalize messy, heterogeneous, or inconsistently formatted data records using agent reasoning.

## Example

```python
from dataclasses import dataclass
from peteos import AgenticObjectBase, tool, invoke_agent

class PersonRecord(AgenticObjectBase):
    def __init__(self):
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
        ...

    @tool
    def normalize_height(self, value: str) -> float:
        """Convert height to centimeters."""
        ...

@dataclass
class CleansingResult:
    name_parts: tuple[str, str, str, str]
    height_cm: float
    birth_iso: str

result = invoke_agent(
    PersonRecord(),
    prompt="Normalize this record: split the name, convert height to cm, "
           "and format the date as ISO 8601.",
    output_schema=CleansingResult,
    thread_id="clean-001",
)
```

**Expected output:**

```python
CleansingResult(
    name_parts=("Dr.", "John", "Michael", "Jr."),
    height_cm=180.34,
    birth_iso="1990-03-15",
)
```

## Why an agent?

| Problem | Why parsing fails | What the agent does |
|---|---|---|
| Height as "5'11"", "180cm", "1.8m" | No single format to parse | Understands each value, normalizes |
| Name with title, middle, suffix | Can't split reliably | Recognizes semantic parts |
| Dates in various formats | No parser handles all conventions | Recognizes, converts to standard |
