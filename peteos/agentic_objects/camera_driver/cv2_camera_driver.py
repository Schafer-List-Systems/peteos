"""CV2CameraDriver - portable V4L2 camera driver via OpenCV.

Unlike OpenCVCameraDriver, this driver does not shell out to
v4l2-ctl. It simply tests indices 0-9 with OpenCV, making it
portable across Linux, Windows, and macOS.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from peteos.agentic_objects.camera_driver import CameraDriver

if TYPE_CHECKING:
    pass


class CV2CameraDriver(CameraDriver):
    """Portable V4L2 driver backed by OpenCV only."""

    driver_name = "opencv-cv2"

    def list_cameras(self) -> list[tuple[int, str]]:
        cameras: list[tuple[int, str]] = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, ""))
                cap.release()
        return cameras

    def open(self, camera_id: int) -> bool:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return False
        self._cap = cap
        return True

    def close(self) -> None:
        if hasattr(self, "_cap"):
            self._cap.release()
            del self._cap

    def grab_frame(self) -> tuple[bool, np.ndarray, tuple[int, int]]:
        if not hasattr(self, "_cap"):
            return False, np.empty((0, 0, 3)), (0, 0)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, np.empty((0, 0, 3)), (0, 0)
        return True, frame, (frame.shape[1], frame.shape[0])