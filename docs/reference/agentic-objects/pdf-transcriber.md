# PdfTranscriber

An agentic object that transcribes PDF pages into text and image descriptions using OCR and LLM vision.

## Description

`PdfTranscriber` is an agentic object addressed to the user. The user calls `transcribe()` with a path to a PDF file and receives structured output containing text transcriptions and image descriptions for each page.

The transcriber produces high-quality results by combining two text sources:
1. **LLM vision** — text transcription and image description from a multimodal LLM.
2. **Tesseract OCR** — text recognition via tesseract.

When cross-referencing is enabled, both sources are unified to produce the most complete and accurate result.

## Usage

```python
transcriber = PdfTranscriber()
results = await transcriber.transcribe(
    "document.pdf",
    cross_reference=True
)
for page_num, (text, description) in enumerate(results, start=1):
    print(f"--- Page {page_num} ---")
    print(f"Text: {text}")
    print(f"Description: {description}")
```

### `transcribe(pdf_path: str, cross_reference: bool = False) -> list[tuple[str, str]]`

Transcribe a multi-page PDF.

**Args:**
- `pdf_path` — Path to the PDF file.
- `cross_reference` — When `True`, cross-reference tesseract and LLM output per page. When `False`, use tesseract only (no LLM needed).

**Returns:** A list of `(page_text, image_description)` pairs, one per page.

**Process:**
1. Splits the PDF into single-page files using `mutool`.
2. Converts each page to a PNG image.
3. Runs tesseract OCR for text transcription.
4. If cross-referencing is enabled, runs the LLM vision agent for transcription and image description, then cross-references both sources.
5. If cross-referencing is disabled, uses the tesseract output directly.

**Requirements:**
- `mutool` (from mupdf) for PDF splitting and rendering.
- `tesseract-ocr` for OCR.
- An LLM backend configured via `ChatBotManager` (required only when `cross_reference=True`).

## Internal Agentic Behavior

When `cross_reference=True`, the transcriber internally invokes its own agent twice per page:
- Once for text transcription and image description using the page image.
- Once for cross-referencing tesseract and LLM outputs into a unified result.

These internal invocations use `persistent_thread_id=None` (transient).
