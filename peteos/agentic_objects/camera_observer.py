"""CameraObserver - webcam access via a CameraDriver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2

from peteos.agentic_objects.camera_driver import CameraDriver, CV2CameraDriver
from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

if TYPE_CHECKING:
    from peteos.session import Session


class CameraObserver(AgenticObject):
    """You are an observer with access to cameras.

    Only one camera can be open at a time. Grabbed images can be read using read_cached_image.
    In Python code, you can access the image's raw PNG bytes via this._cached_camera_frame.
    """

    def __init__(
        self,
        driver: CameraDriver | None = None,
        scaling: int | None = None,
    ) -> None:
        super().__init__()
        self._driver: CameraDriver = driver or CV2CameraDriver()
        self._scaling: int | None = scaling
        self._active_camera_id: int | None = None
        self._cached_camera_frame: bytes | None = None
        self._cached_shape: tuple[int, int] | None = None

    def _encode_frame(self, frame: cv2.Mat) -> tuple[bool, bytes, tuple[int, int]]:
        """Encode a numpy frame as PNG bytes."""
        _, encoded = cv2.imencode(".png", frame)
        return True, encoded.tobytes(), (frame.shape[1], frame.shape[0])

    def _resize_if_needed(self, frame: cv2.Mat) -> cv2.Mat:
        """Resize frame if scaling is configured."""
        if self._scaling is None or self._scaling <= 0:
            return frame
        h, w = frame.shape[:2]
        if w > h:
            new_w = self._scaling
            new_h = round(h * self._scaling / w)
        else:
            new_h = self._scaling
            new_w = round(w * self._scaling / h)
        return cv2.resize(frame, (new_w, new_h))

    def _clear_state(self) -> None:
        """Clear active camera state and cache."""
        self._active_camera_id = None
        self._cached_camera_frame = None
        self._cached_shape = None

    @tool(description="List all available cameras.")
    def list_cameras(self) -> str:
        cameras = self._driver.list_cameras()
        if not cameras:
            return "No cameras found."
        return "Available cameras:\n" + "\n".join(
            f"  Camera ID: {cid}, Driver: {self._driver.driver_name}, Description: {name}"
            for cid, name in cameras
        )

    @tool(description="Open a camera by ID. Must be called before grab_image.")
    def open_camera(self, camera_id: int) -> str:
        if self._active_camera_id is not None:
            return "ERROR: A camera is already open. Call close_camera first."

        ok = self._driver.open(camera_id)
        if not ok:
            return f"ERROR: Camera {camera_id} could not be opened."

        self._active_camera_id = camera_id

        # Warm up: grab a frame to warm up the stream.
        ok, frame, _ = self._driver.grab_frame()
        if not ok or frame is None:
            self._driver.close()
            self._clear_state()
            return f"ERROR: Camera {camera_id} warm-up failed."

        return f"OK: Camera {camera_id} (driver: {self._driver.driver_name}) opened."

    @tool(description="Close the currently open camera.")
    def close_camera(self) -> str:
        if self._active_camera_id is None:
            return "No camera is currently open."

        self._driver.close()
        self._clear_state()
        return "OK: Camera closed."

    @tool(description="Capture a single image from the open camera. Caches the frame in memory.")
    def grab_image(self) -> str:
        if self._active_camera_id is None:
            return "ERROR: No camera is open."

        ok, frame, _ = self._driver.grab_frame()
        if not ok or frame is None:
            return "ERROR: Failed to capture image."

        frame = self._resize_if_needed(frame)

        ok, encoded, (w, h) = self._encode_frame(frame)
        if not ok:
            return "ERROR: Failed to encode image."

        self._cached_camera_frame = encoded
        self._cached_shape = (w, h)
        return f"OK: Image captured ({w}x{h}), cached in memory."

    @tool(description="Read the cached image from the last grab_image call and send it to the session.")
    async def read_cached_image(self, runner: "Runner") -> str:
        if self._cached_camera_frame is None:
            return "ERROR: No image has been captured yet. Call grab_image first."

        if runner is None:
            return "ERROR: Runner not available."

        await self._send_media(
            data=self._cached_camera_frame,
            mime_type="image/png",
            runner=runner,
        )
        return "OK: Image sent to session."
