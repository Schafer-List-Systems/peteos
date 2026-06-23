"""OAP benchmarks for AgenticStringComparator.

Tests the correctness of the LLM-based comparison functions
(contains, is_same, contradicts, is_complete) using known-good
input/output pairs with Monte Carlo iterations for statistical
confidence.
"""

from __future__ import annotations

import pytest

from peteos.oap.benchmark import AgenticStringComparator, BenchmarkReport, BenchmarkRow, BenchmarkRunner


@pytest.mark.oap
async def test_contains():
    """Benchmark AgenticStringComparator.contains with known-positive and known-negative pairs."""
    comp = AgenticStringComparator.instance()

    test_cases = [
        (True, "The sky is blue", "The color of the world's ceiling"),
        (True, "Water boils at 100 degrees Celsius", "Boiling temperatur of water"),
        (True, "The plant uses sunlight to produce energy", "energy production"),
        (False, "The sky is blue", "green"),
        (False, "Water boils at 100 degrees Celsius", "freezing temperature"),
        (False, "Photosynthesis converts sunlight into energy", "nuclear energy"),
    ]

    async def test_fn(row: BenchmarkRow) -> bool:
        expected, text, substring = row.input_dimensions["test_case"]
        result = await comp.contains(text, substring)
        return result == expected

    runner = BenchmarkRunner(test_fn)
    runner.add_dimension("test_case", test_cases)
    runner.add_dimension("run", range(5))

    report: BenchmarkReport = await runner.run()

    summary = report.average(["success"])
    assert summary["success"] >= 0.8, f"contains accuracy: {summary}"
