# Reference

An index of all reference pages in this directory. Each page documents a single class or module completely.

## Core Classes

| Reference | Description |
|---|---|
| [AgenticObjectBase](./agentic-object-base.md) | Base class for all agentic objects |
| [Session](./session.md) | Interaction thread with context |
| [Context](./context.md) | Mutable chat history, forking, and compaction |
| [Message](./message.md) | Message and ContentPart wrappers |

## Configuration

| Reference | Description |
|---|---|
| [Decorator Arguments](./decorator-args.md) | `@agentic_object` and `@tool` options |

## Media

| Reference | Description |
|---|---|
| [ContentMedia](./content-media.md) | Attaching images, video, and PDFs |

## Error Handling

| Reference | Description |
|---|---|
| [Error](./error.md) | `Error` value object returned by `invoke_agent` |

## Concrete Agentic Objects

| Reference | Description |
|---|---|
| [BashWorkspace](./agentic-objects/bash-workspace.md) | Secure sandbox for executing shell commands |
| [CameraObserver](./agentic-objects/camera-observer.md) | Webcam access via a pluggable camera driver |
| [PdfTranscriber](./agentic-objects/pdf-transcriber.md) | High-quality PDF transcription via tesseract + LLM vision |
| [TextEditor](./agentic-objects/text-editor.md) | Line-based text editing with mtime-based safety |
| [WebNavigator](./agentic-objects/web-navigator.md) | Web fetching, rendering, and screenshot capture |
