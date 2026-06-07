"""Statistical LLM Benchmarking for OAP.

Provides BenchmarkRunner, BenchmarkRow, BenchmarkReport, and
LLM-based comparison functions for statistical evaluation of
AgenticObjectBase instances.
"""

from __future__ import annotations

import statistics
import sys
from itertools import product
from typing import TYPE_CHECKING, Any, Callable

from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool
from peteos.oap.error import Error

if TYPE_CHECKING:
    from peteos.session import Session


class BenchmarkRow:
    """A single row in a benchmark result table.

    Attributes:
        input_dimensions: Set by BenchmarkRunner (immutable after creation).
        output_dimensions: Written by the test function (arbitrary key/value pairs).
    """

    def __init__(self, input_dimensions: dict[str, Any]) -> None:
        self.input_dimensions = input_dimensions
        self.output_dimensions: dict[str, Any] = {}


class BenchmarkReport(AgenticObjectBase):
    """Agentic report over benchmark results.

    Inherits AgenticObjectBase so it has its own agent for invoke_agent()
    and reason() via invoke_agent. Also provides filter and aggregation methods.

    Attributes:
        input_dimensions: List of dimension names.
        rows: All collected BenchmarkRow instances.
    """

    def __init__(self, input_dimensions: list[str], rows: list[BenchmarkRow]) -> None:
        super().__init__()
        self.input_dimensions = input_dimensions
        self.rows = rows

    def filter(self, predicate: Callable[[BenchmarkRow], bool]) -> BenchmarkReport:
        """Return a new BenchmarkReport containing only rows matching the predicate."""
        return BenchmarkReport(self.input_dimensions, [r for r in self.rows if predicate(r)])

    def _collect_values(self, columns: list[str], row_subset: list[BenchmarkRow] | None) -> dict[str, list[Any]]:
        """Collect values for the specified columns from the row subset."""
        subset = row_subset or self.rows
        result: dict[str, list[Any]] = {col: [] for col in columns}
        for row in subset:
            for col in columns:
                if col in row.output_dimensions:
                    result[col].append(row.output_dimensions[col])
        return result

    def max(self, columns: list[str], rows: list[BenchmarkRow] | None = None) -> dict[str, Any]:
        """Maximum value per column over the given rows."""
        values = self._collect_values(columns, rows)
        return {col: max(vals) for col, vals in values.items() if vals}

    def average(self, columns: list[str], rows: list[BenchmarkRow] | None = None) -> dict[str, float]:
        """Arithmetic mean per column over the given rows."""
        values = self._collect_values(columns, rows)
        return {col: float(statistics.mean(vals)) for col, vals in values.items() if vals}

    def median(self, columns: list[str], rows: list[BenchmarkRow] | None = None) -> dict[str, float]:
        """Median value per column over the given rows."""
        values = self._collect_values(columns, rows)
        return {col: float(statistics.median(vals)) for col, vals in values.items() if vals}

    def summarize(self, columns: list[str], rows: list[BenchmarkRow] | None = None) -> str:
        """LLM-based natural-language summary of the values in the given columns."""
        values = self._collect_values(columns, rows)
        summary_parts = []
        for col, vals in values.items():
            summary_parts.append(f"### {col}")
            if vals:
                summary_parts.append(f"- Count: {len(vals)}")
                summary_parts.append(f"- Values: {repr(vals)}")
        summary_text = "\n".join(summary_parts)
        return self.invoke_agent(
            f"Summarize the following benchmark data in natural language.\n\n{summary_text}\n\n"
            "Provide a concise interpretation of what these values mean.\n"
            "Focus on patterns, outliers, and meaningful observations.",
            persistent_thread_id=None,
        )

    @tool(name="list_columns", description="List the names of all available output dimensions (columns) in the report.")
    def list_columns(self) -> list[str]:
        """Return all output column names."""
        columns: set[str] = set()
        for row in self.rows:
            columns.update(row.output_dimensions.keys())
        return sorted(columns)

    @tool(name="agg_max", description="Compute the maximum value for each of the specified columns across all rows.")
    def agg_max(self, columns: str) -> dict[str, Any]:
        """Maximum per column. Columns is a comma-separated string of column names."""
        return self.max([c.strip() for c in columns.split(",")])

    @tool(name="agg_average", description="Compute the arithmetic mean for each of the specified columns across all rows.")
    def agg_average(self, columns: str) -> dict[str, float]:
        """Average per column. Columns is a comma-separated string of column names."""
        return self.average([c.strip() for c in columns.split(",")])

    @tool(name="agg_median", description="Compute the median for each of the specified columns across all rows.")
    def agg_median(self, columns: str) -> dict[str, float]:
        """Median per column. Columns is a comma-separated string of column names."""
        return self.median([c.strip() for c in columns.split(",")])

    @tool(name="agg_summarize", description="Generate a natural-language summary of the values in the specified columns.")
    def agg_summarize(self, columns: str) -> str:
        """LLM-based summary of column values. Columns is a comma-separated string."""
        return self.summarize([c.strip() for c in columns.split(",")])


class BenchmarkRunner:
    """Minimal benchmark runner: Cartesian product loop and row collection.

    Prints per-row progress, running accuracy, and a final aggregate summary
    including per-dimension breakdowns.

    Does NOT create target objects, count tokens, evaluate thresholds, or
    aggregate metrics. All of that is the developer's responsibility.

    Args:
        test_fn: Async callback called once per Cartesian product combination.
            Must return a boolean: True for success, False for failure.
    """

    def __init__(self, test_fn: Callable[[BenchmarkRow], Any]) -> None:
        self._test_fn = test_fn
        self._dimensions: dict[str, list[Any]] = {}

    def add_dimension(self, name: str, values: list[Any] | range) -> None:
        """Add an input dimension with the given name and values."""
        self._dimensions[name] = list(values)

    def _format_value(self, v: Any) -> str:
        """Format a dimension value for display."""
        if isinstance(v, str) and len(v) > 40:
            return v[:40] + "..."
        return repr(v)

    def _print_separator(self) -> None:
        print("-" * 80, flush=True, file=sys.stderr)

    def _print_progress_line(
        self, row_num: int, total: int, input_dims: dict[str, Any]
    ) -> None:
        """Print per-row progress with input dimension values."""
        dim_strs = ", ".join(f"{k}={self._format_value(v)}" for k, v in input_dims.items())
        print(f"  [{row_num}/{total}] {dim_strs}", flush=True, file=sys.stderr)

    def _print_result_line(
        self, row_num: int, total: int, input_dims: dict[str, Any], output_dims: dict[str, Any]
    ) -> None:
        """Print per-row input dimensions, result, and running success rate."""
        success = output_dims.get("success", True)
        success_bool = success if isinstance(success, bool) else True
        # Update running stats
        if not hasattr(self, "_success_count"):
            self._success_count = 0
            self._success_total = 0
        self._success_total += 1
        if success_bool:
            self._success_count += 1
        pass_count = self._success_count
        pass_rate = pass_count / self._success_total * 100
        status = f"[{pass_count}/{self._success_total} ({pass_rate:.1f}%)]"

        input_str = ", ".join(f"{k}={self._format_value(v)}" for k, v in input_dims.items())
        output_str = ", ".join(f"{k}={self._format_value(v)}" for k, v in output_dims.items())
        print(f"    {row_num:>4}/{total}  {input_str} → {output_str} {status}", flush=True, file=sys.stderr)

    def _print_final_summary(self, rows: list[BenchmarkRow], total: int) -> None:
        """Print final aggregate summary with per-dimension breakdowns."""
        self._print_separator()
        print("FINAL SUMMARY", file=sys.stderr)
        self._print_separator()

        # Overall success rate
        if rows:
            pass_count = sum(1 for r in rows if r.output_dimensions.get("success", True))
            pass_rate = pass_count / len(rows) * 100
            print(f"  Overall:  {pass_count}/{len(rows)} ({pass_rate:.1f}%)", file=sys.stderr)

        # Per-dimension breakdown (only "success" field)
        if self._dimensions:
            dim_names = list(self._dimensions.keys())
            for dim_name in dim_names:
                for dim_val in self._dimensions[dim_name]:
                    subset = [
                        r for r in rows
                        if dim_val in r.input_dimensions.values()
                        or (isinstance(r.input_dimensions.get(dim_name), type(dim_val))
                            and r.input_dimensions.get(dim_name) == dim_val)
                    ]
                    if subset:
                        subset_pass = sum(1 for r in subset if r.output_dimensions.get("success", True))
                        subset_rate = subset_pass / len(subset) * 100
                        dim_str = self._format_value(dim_val)
                        print(f"  {dim_name}={dim_str}: {subset_pass}/{len(subset)} ({subset_rate:.1f}%)", file=sys.stderr)

        # Print rows with failures
        failed_rows = [
            r for r in rows
            if not isinstance(r.output_dimensions.get("success", True), bool)
            or r.output_dimensions.get("success", True) is False
        ]

        if failed_rows:
            print(f"\n  {len(failed_rows)} failed row(s):", file=sys.stderr)
            for row in failed_rows[:20]:
                dim_str = ", ".join(f"{k}={self._format_value(v)}" for k, v in row.input_dimensions.items())
                result_str = ", ".join(f"{k}={self._format_value(v)}" for k, v in row.output_dimensions.items())
                print(f"    INPUT:  {dim_str}", file=sys.stderr)
                print(f"    OUTPUT: {result_str}", file=sys.stderr)

        self._print_separator()

    async def run(self) -> BenchmarkReport:
        """Execute the benchmark and return a BenchmarkReport.

        Prints per-row progress, running accuracy, and a final aggregate
        summary including per-dimension breakdowns.
        """
        rows: list[BenchmarkRow] = []
        if not self._dimensions:
            row = BenchmarkRow({})
            success = await self._test_fn(row)
            row.output_dimensions["success"] = success
            rows.append(row)
        else:
            dim_names = list(self._dimensions.keys())
            dim_values = [self._dimensions[n] for n in dim_names]
            total = 1
            for values in dim_values:
                total *= len(values)

            combinations_iter = product(*dim_values)

            for idx, combination in enumerate(combinations_iter, 1):
                input_dims = dict(zip(dim_names, combination))
                row = BenchmarkRow(input_dims)
                success = await self._test_fn(row)
                row.output_dimensions["success"] = success
                rows.append(row)
                self._print_result_line(idx, total, input_dims, row.output_dimensions)

        self._print_final_summary(rows, len(rows))
        return BenchmarkReport(dim_names if dim_names else [], rows)


# --- LLM-based comparison functions ---


class AgenticStringComparator(AgenticObjectBase):
    """You are a String Comparator answering questions with true or false.
    You provide your answer by calling the produce_output tool with a boolean value as argument.
    """

    _instance: AgenticStringComparator | None = None

    @classmethod
    def instance(cls) -> AgenticStringComparator:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def _question(self, prompt: str) -> bool:
        """Ask the judge a yes/no question, returning bool via produce_output."""
        result = await self.invoke_agent(
            prompt,
            output_schema=bool,
            persistent_thread_id=None,
        )
        if isinstance(result, Error):
            raise result
        return bool(result)

    async def contains(self, text: str, substring: str) -> bool:
        """Does the text contain this information? (LLM-evaluated semantic containment.)"""
        return await self._question(
            f"Does the following text contain the information {substring!r}?\n\nText: {text}"
        )

    async def is_same(self, a: str, b: str) -> bool:
        """Are these two texts semantically equivalent?"""
        return await self._question(
            "Are these two texts semantically equivalent (they express the same meaning)?\n\n"
            f"Text A: {a!r}\n\nText B: {b!r}"
        )

    async def contradicts(self, text: str, claim: str) -> bool:
        """Does the text contradict the claim?"""
        return await self._question(
            f"Does the following text contradict the claim {claim!r}?\n\nText: {text}"
        )

    async def is_complete(self, text: str, required_info: str) -> bool:
        """Does the text contain all required information?"""
        return await self._question(
            f"Does the following text contain the required information: {required_info!r}?\n\nText: {text}"
        )
