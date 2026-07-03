# Reference

An index of all reference pages in this directory. Each page documents a single class or module completely.

## Core API

| Reference | Description |
|---|---|
| [AgenticObject](./agentic-object-base.md) | Base class for all agentic objects |
| [ChatBotManager](./chatbot-manager.md) | Configure LLM backends |
| [Error](./error.md) | Value object returned by `invoke_agent` on failure |
| [Role](./role.md) | User-overridable configuration and identity for agentic objects |
| [RoleManager](./rolemanager.md) | Load, merge, and persist role overrides |

## Configuration

| Reference | Description |
|---|---|
| [Decorator Arguments](./decorator-args.md) | `@agentic_object` and `@tool` options |

## Conversation Primitives

| Reference | Description |
|---|---|
| [Message & ContentPart](./message.md) | Message and ContentPart factories — for steering the agent |

## Concrete Agentic Objects

| Reference | Description |
|---|---|
| [BashWorkspace](./agentic-objects/bash-workspace.md) | Secure sandbox for executing shell commands |
| [CameraObserver](./agentic-objects/camera-observer.md) | Webcam access via a pluggable camera driver |
| [PdfTranscriber](./agentic-objects/pdf-transcriber.md) | High-quality PDF transcription via tesseract + LLM vision |
| [TextEditor](./agentic-objects/text-editor.md) | Line-based text editing with mtime-based safety |
| [WebNavigator](./agentic-objects/web-navigator.md) | Web fetching, rendering, and screenshot capture |
