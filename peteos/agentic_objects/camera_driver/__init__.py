"""CameraDriver - abstract interface for camera backends."""

from __future__ import annotations

from peteos.agentic_objects.camera_driver.camera_driver import CameraDriver
from peteos.agentic_objects.camera_driver.cv2_camera_driver import CV2CameraDriver

__all__ = ["CameraDriver", "CV2CameraDriver"]