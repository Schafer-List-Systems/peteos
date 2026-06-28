"""CameraDriver - abstract interface for camera backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraDriver(ABC):
    """Interface for camera backends.

    A driver is responsible for listing, opening, closing, and grabbing
    frames from a specific camera subsystem (e.g. V4L2, USB, RTSP).
    The driver returns raw numpy arrays; encoding/scaling is handled
    by the observer.
    """

    driver_name: str = ""

    @abstractmethod
    def list_cameras(self) -> list[tuple[int, str]]:
        """List available cameras.

        Returns:
            List of (camera_id, description) tuples.
        """

    @abstractmethod
    def open(self, camera_id: int) -> bool:
        """Open a camera for capturing.

        Args:
            camera_id: The camera ID from ``list_cameras``.

        Returns:
            True if the camera was opened successfully.
        """

    @abstractmethod
    def close(self) -> None:
        """Close the currently open camera."""

    @abstractmethod
    def grab_frame(self) -> tuple[bool, np.ndarray, tuple[int, int]]:
        """Grab a single raw frame.

        Returns:
            Tuple of (success, numpy array, (width, height)).
        """