"""Agentic Objects - concrete implementations using the OAP framework."""

from peteos.agentic_objects.camera_driver import CameraDriver, CV2CameraDriver
from peteos.agentic_objects.camera_observer import CameraObserver
from peteos.agentic_objects.multi_camera_observer import MultiCameraObserver
from peteos.agentic_objects.pdf_transcriber import PdfTranscriber
from peteos.agentic_objects.bash_workspace import BashWorkspace
from peteos.agentic_objects.text_editor import TextEditor

__all__ = [
    "BashWorkspace",
    "CameraDriver",
    "CV2CameraDriver",
    "CameraObserver",
    "MultiCameraObserver",
    "PdfTranscriber",
    "TextEditor",
]