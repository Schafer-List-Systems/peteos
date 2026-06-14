"""MultiCameraObserver - manages multiple camera drivers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from peteos.agentic_objects.camera_driver import CameraDriver
from peteos.oap.base import AgenticObjectBase
from peteos.oap.decorators import tool

if TYPE_CHECKING:
    from peteos.session import Session


class MultiCameraObserver(AgenticObjectBase):
    """Observer with access to multiple camera drivers.

    Registers drivers by name. When listing cameras, returns
    the driver name, camera ID, and description for all drivers.
    Only one camera (from any driver) can be open at a time.
    """

    def __init__(self, scaling: int | None = None) -> None:
        super().__init__()
        self._drivers: dict[str, CameraDriver] = {}
        self._scaling: int | None = scaling
        self._active_driver: CameraDriver | None = None
        self._active_camera_id: int | None = None
        self._cached_frame: bytes | None = None
        self._cached_shape: tuple[int, int] | None = None

    def register_driver(self, driver: CameraDriver) -> None:
        """Register a camera driver by name."""
        self._drivers[driver.driver_name] = driver

    @tool(description="List all available cameras across all registered drivers.")
    def list_cameras(self) -> str:
        cameras: list[tuple[str, int, str]] = []
        for driver in self._drivers.values():
            for cid, desc in driver.list_cameras():
                cameras.append((driver.driver_name, cid, desc))
        if not cameras:
            return "No cameras found."
        lines = []
        for driver_name, cid, desc in cameras:
            lines.append(
                f"  Driver: {driver_name}, Camera ID: {cid}, Description: {desc}"
            )
        return "Available cameras:\n" + "\n".join(lines)

    @tool(description="Open a camera from a specific driver.")
    def open_camera(self, driver_name: str, camera_id: int) -> str:
        if self._active_driver is not None:
            return "ERROR: A camera is already open. Call close_camera first."

        driver = self._drivers.get(driver_name)
        if driver is None:
            return f"ERROR: Unknown driver '{driver_name}'. Register it first."

        if not driver.open(camera_id):
            return f"ERROR: Camera {camera_id} on driver '{driver_name}' could not be opened."

        self._active_driver = driver
        self._active_camera_id = camera_id
        self._cached_frame = None
        self._cached_shape = None

        # Warm up: grab a frame.
        ok, frame, _ = driver.grab_frame()
        if not ok or frame is None:
            driver.close()
            self._active_driver = None
            self._active_camera_id = None
            self._cached_frame = None
            self._cached_shape = None
            return f"ERROR: Camera warm-up failed on '{driver_name}'."

        return f"OK: Camera {camera_id} opened (driver: {driver_name})."

    @tool(description="Close the currently open camera.")
    def close_camera(self) -> str:
        if self._active_driver is None:
            return "No camera is currently open."

        self._active_driver.close()
        self._active_driver = None
        self._active_camera_id = None
        self._cached_frame = None
        self._cached_shape = None
        return "OK: Camera closed."

    @tool(description="Capture a single image from the open camera. Caches the frame in memory.")
    def grab_image(self) -> str:
        if self._active_driver is None:
            return "ERROR: No camera is open."

        ok, frame, (raw_w, raw_h) = self._active_driver.grab_frame()
        if not ok or frame is None:
            return "ERROR: Failed to capture image."

        # Resize if scaling is configured.
        if self._scaling is not None and self._scaling > 0:
            h, w = frame.shape[:2]
            if w > h:
                new_w = self._scaling
                new_h = round(h * self._scaling / w)
            else:
                new_h = self._scaling
                new_w = round(w * self._scaling / h)
            frame = cv2.resize(frame, (new_w, new_h))

        # Encode as PNG.
        _, encoded = cv2.imencode(".png", frame)
        self._cached_frame = encoded.tobytes()
        self._cached_shape = (frame.shape[1], frame.shape[0])
        return f"OK: Image captured ({self._cached_shape[0]}x{self._cached_shape[1]}), cached in memory."

    @tool(description="Read the cached image from the last grab_image call and send it to the session.")
    async def read_cached_image(self, session: "Session | None" = None) -> str:
        if self._cached_frame is None:
            return "ERROR: No image has been captured yet. Call grab_image first."

        if session is None:
            return "ERROR: Session not available."

        await self._send_media(
            data=self._cached_frame,
            mime_type="image/png",
            session=session,
        )
        return "OK: Image sent to session."
