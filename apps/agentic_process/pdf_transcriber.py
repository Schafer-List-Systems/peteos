"""PdfTranscriber - high-quality PDF transcription via tesseract + LLM vision."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from peteos.oap.base import AgenticObjectBase, Error


class PdfTranscriber(AgenticObjectBase):
    """You are a PDF page transcriber.

    Given a single-page PDF image, produce:
    1. A clean text transcription of all text content on the page.
       Preserve the logical reading order and formatting.
    2. A brief description of any images, tables, charts, or diagrams
       visible on the page. If there are no visual elements beyond text,
       return an empty string for the description.

    When transcribing, preserve headings, labels, numbers, and units
    exactly as they appear. Do not add commentary or interpretation.

    When describing visual elements, state what type of element it is
    (image, chart, diagram, table), where it is located on the page,
    and what it depicts. For tables, transcribe the content under "text"
    and only summarize the visual representation under "description".
    """

    async def transcribe(self, pdf_path: str, cross_reference = False) -> list[tuple[str, str]]:
        """Transcribe a multi-page PDF.

        Splits the PDF into single-page files, then for each page:
        1. Converts to PNG via mutool
        2. Runs LLM vision agent for text transcription + image description
        3. Runs tesseract OCR for cross-reference
        4. Runs cross-reference agent to unify both text sources
        5. Returns (unified_text, image_description) pair

        Args:
            pdf_path: Path to the PDF file.
            cross_reference: cross reference tesseract and LLM output per Page, when true. Otherwise, use tesseract only

        Returns:
            A list of (page_text, image_description) pairs, one per page.
        """
        tmpdir = tempfile.mkdtemp(prefix="pdf_transcribe_")
        tmpdir_path = Path(tmpdir)

        try:
            self._split_pdf(pdf_path, tmpdir_path)
            results: list[tuple[str, str]] = []
            pages = sorted(tmpdir_path.glob("page-*.pdf"))
            for page_pdf in pages:
                page_name = page_pdf.stem  # e.g. page-1
                png_path = tmpdir_path / f"{page_name}.png"
                self._render_page(page_pdf, png_path)
                tesseract_text = self._tesseract_transcribe(str(png_path))
                if cross_reference:
                    agent_result = await self._vision_agent_transcribe(str(png_path))
                    image_description = agent_result.image_description
                    unified_text = await self._cross_reference(
                        tesseract_text,
                        agent_result.text,
                    )
                else:
                    unified_text = tesseract_text
                    image_description = ""
                page_pdf.unlink(missing_ok=True)
                png_path.unlink(missing_ok=True)
                results.append((unified_text, image_description))
            return results
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _split_pdf(pdf_path: str, tmpdir: Path) -> None:
        """Split PDF into single-page PDFs using mutool.

        Uses pdfinfo to count pages, then mutool merge with page number
        to extract each page individually.
        """
        info = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        num_pages = None
        for line in info.stdout.splitlines():
            if line.startswith("Pages:"):
                num_pages = int(line.split(":")[1].strip())
                break
        if num_pages is None:
            raise ValueError("Could not determine page count from pdfinfo")
        for i in range(1, num_pages + 1):
            name = f"page-{i:03d}"
            subprocess.run(
                ["mutool", "merge", "-o", str(tmpdir / f"{name}.pdf"),
                 "-O", "garbage=compact", pdf_path, str(i)],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )

    @staticmethod
    def _render_page(page_pdf: Path, png_path: Path) -> None:
        """Convert a single-page PDF to PNG using mutool."""
        subprocess.run(
            ["mutool", "draw", "-F", "png", "-w", "1600", "-r", "150",
             "-o", str(png_path), str(page_pdf), "1"],
            check=True,
            timeout=30,
        )

    @staticmethod
    def _tesseract_transcribe(png_path: str) -> str:
        """Transcribe a PNG image using tesseract OCR."""
        try:
            result = subprocess.run(
                ["tesseract", png_path, "-", "--psm", "6"],
                capture_output=True,
                text=True,
                check=True,
                timeout=60,
            )
            return result.stdout
        except FileNotFoundError:
            return ""
        except subprocess.TimeoutExpired:
            return ""
        except Exception:
            return ""

    async def _vision_agent_transcribe(self, png_path: str) -> "TranscriptionResult":
        """Run the LLM vision agent for text transcription + image description."""
        prompt = (
            "Transcribe the text content of this PDF page exactly as it appears. "
            "Also describe ALL images, tables, charts, or diagrams visible on the page.\n\n"
        )
        output = await self.invoke_agent(
            prompt=prompt,
            image=png_path,
            output_schema=TranscriptionResult,
            persistent_thread_id=None,
            timeout=None,
        )
        if isinstance(output, Error):
            raise RuntimeError(output.message)
        return output

    async def _cross_reference(self, tesseract_text: str, vision_text: str) -> str:
        """Cross-reference tesseract and LLM vision transcriptions.

        Produces a single unified text by reconciling both sources.
        """
        prompt = (
            f"Tesseract OCR result:\n{tesseract_text}\n\n"
            f"LLM vision transcription:\n{vision_text}\n\n"
            "Cross-reference both sources and produce a single unified "
            "transcription. Preserve all text that appears in either source. "
            "If the two disagree, prefer the more complete and accurate version. "
            "Return only the unified text, nothing else. "
            "Use the `produce_output` tool to return your result."
        )
        result = await self.invoke_agent(
            prompt=prompt,
            output_schema=UnifiedText,
            persistent_thread_id=None,
            timeout=None,
        )
        if isinstance(result, Error):
            raise RuntimeError(result.message)
        return result.text


@dataclass
class TranscriptionResult:
    """Structured output from the vision agent transcription step.

    Attributes:
        text: The full text transcription of the page.
        image_description: Description of any visual elements on the page.
    """

    text: str
    image_description: str


@dataclass
class UnifiedText:
    """Structured output from the cross-reference step.

    Attributes:
        text: The unified transcription from cross-referencing tesseract and LLM sources.
    """

    text: str
