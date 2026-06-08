"""OAP benchmark for PdfTranscriber.

Measures the reliability and completeness of PDF transcription
using the AgenticStringComparator as judge.
"""

from __future__ import annotations

import pytest

from peteos.oap.benchmark import AgenticStringComparator, BenchmarkReport, BenchmarkRow, BenchmarkRunner


@pytest.mark.oap
async def test_pdf_transcription():
    """Benchmark PdfTranscriber with sample PDF files.

    This is a template benchmark — requires actual PDF test files to be present.
    """
    comp = AgenticStringComparator.instance()

    # TODO: Replace with actual test PDF data
    test_files: list[tuple[str, str]] = []

    async def test_fn(row: BenchmarkRow) -> None:
        file_path, expected_info = row.input_dimensions["file"]
        # TODO: Implement transcriber creation and invocation
        # obj = PdfTranscriber()
        # result = await obj.transcribe(file_path)
        # found = await comp.contains(result, expected_info)
        row.output_dimensions["placeholder"] = True

    runner = BenchmarkRunner(test_fn)
    runner.add_dimension("file", test_files)
    runner.add_dimension("run", range(5))

    report: BenchmarkReport = await runner.run()
    assert report.rows, "Expected at least one row"
