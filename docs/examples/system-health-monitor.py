"""Example: System Health Monitor — agentic object with tools, sandbox code, and structured output.

Usage: python system-health-monitor.py <backend-url>
"""

import asyncio
import sys
from dataclasses import dataclass, field
from enum import Enum

from peteos.chatbot.manager import ChatBotManager
from peteos.oap.base import AgenticObject
from peteos.oap.decorators import agentic_object, tool


class AlertLevel(Enum):
    OK = "OK"
    WARNING = "Warning"
    CRITICAL = "Critical"


@dataclass
class Metric:
    name: str
    current_value: float
    unit: str
    history: list[float] = field(default_factory=list)


@agentic_object(allow_code_execution=True)
class SystemHealthMonitor(AgenticObject):
    """You are a system health monitor. You track CPU, memory, disk, and
    network metrics. Use record_metric to add readings and get_metric to
    inspect current values. The agent can analyze trends and return a
    structured health report using generate_report."""

    def __init__(self):
        super().__init__()
        self._metrics: dict[str, Metric] = {}

    @tool
    def record_metric(self, name: str, value: float, unit: str, history: list[float]) -> str:
        """Record a metric reading with its current value, unit, and historical data."""
        self._metrics[name] = Metric(name=name, current_value=value, unit=unit, history=history)
        return f"Recorded {name} = {value} {unit}."

    @tool
    def get_metric(self, name: str) -> dict:
        """Return the current metric and its history."""
        m = self._metrics.get(name)
        if not m:
            return {"error": f"Unknown metric: {name}"}
        return {"name": m.name, "current_value": m.current_value, "unit": m.unit, "history": m.history}

    @tool
    def list_metrics(self) -> list[str]:
        """Return the names of all tracked metrics."""
        return list(self._metrics.keys())


@dataclass
class MetricAlert:
    name: str
    level: AlertLevel
    current_value: float
    message: str


@dataclass
class HealthReport:
    overall_status: AlertLevel
    alerts: list[MetricAlert]
    summary: str


async def main():
    #from peteos.utils.logger import setup_logging
    #setup_logging(level="DEBUG", debug=True)
    args = sys.argv[1:] + [None] * 3
    url, api_type, api_key = args[:3]
    await ChatBotManager.add_backend("local", url, api_type=api_type, api_key=api_key)

    monitor = SystemHealthMonitor()

    # Record metrics
    await monitor.invoke_agent(
        "Record CPU at 87.5% with history [45, 52, 68, 79, 87.5]."
    )
    await monitor.invoke_agent(
        "Record memory at 72.0% with history [60, 65, 68, 70, 72]."
    )
    await monitor.invoke_agent(
        "Record disk at 95.0% with history [70, 75, 80, 88, 95]."
    )

    # Get structured health report
    result = await monitor.invoke_agent(
        "Generate a health report. CPU above 85% is critical, disk above 90% is critical.",
        output_schema=HealthReport,
    )
    print("Health report:", result)

    # Use sandbox to calculate averages
    result = await monitor.invoke_agent(
        "Calculate the average CPU usage over the last 5 readings and predict if it will keep rising.",
        output_schema=MetricAlert,
    )
    print("CPU trend:", result)


if __name__ == "__main__":
    asyncio.run(main())
