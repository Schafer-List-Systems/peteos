# CameraObserver

An agentic object that provides webcam access via a camera driver abstraction.

## Description

`CameraObserver` is an agentic object addressed primarily to the agent, enabling it to capture images from a webcam. However, the user can also interact with it programmatically: by deriving from the camera observer, the user can read the cached image via the `self._cached_camera_frame` attribute, which holds the raw PNG bytes. This allows manipulating the image before sending it to the agent's multimodal LLM.

The observer delegates camera hardware access to a pluggable `CameraDriver` interface. By default it uses `CV2CameraDriver`, a portable driver backed by OpenCV that works across Linux, Windows, and macOS. You can pass a custom driver to the constructor:

```python
observer = CameraObserver(driver=MyCustomDriver())
```

### `CameraDriver` interface

The `CameraDriver` ABC defines the contract for listing, opening, closing, and grabbing frames from a camera. It returns raw numpy arrays; encoding and scaling are handled by the observer.

| Method | Description |
|---|---|
| `list_cameras() -> list[tuple[int, str]]` | List available cameras (camera_id, description). |
| `open(camera_id: int) -> bool` | Open a camera for capturing. |
| `close() -> None` | Close the currently open camera. |
| `grab_frame() -> tuple[bool, np.ndarray, tuple[int, int]]` | Grab a single raw frame (success, numpy array, (width, height)). |

## Tools

### `list_cameras() -> str`

List all available cameras.

**Returns:** A string listing camera IDs, driver names, and descriptions. If no cameras are found, returns `"No cameras found."`

### `open_camera(camera_id: int) -> str`

Open a camera by ID. Must be called before `grab_image`.

**Args:**
- `camera_id` — The camera ID from `list_cameras`.

**Returns:** Confirmation message. Returns an error if a camera is already open or if the camera could not be opened. Only one camera can be open at a time.

### `close_camera() -> str`

Close the currently open camera.

**Returns:** Confirmation message. Returns a message if no camera is currently open.

### `grab_image() -> str`

Capture a single image from the open camera. The frame is cached in memory as PNG bytes.

**Returns:** Confirmation with image dimensions, e.g. `"OK: Image captured (1920x1080), cached in memory."` Returns an error if no camera is open or capture fails.

The captured frame is stored in `self._cached_camera_frame` (raw PNG bytes) and `self._cached_shape` (width, height tuple).

### `read_cached_image(session: Session | None = None) -> str`

Read the cached image from the last `grab_image` call and send it to the session for the LLM to process.

**Args:**
- `session` — The session to send the image to.

**Returns:** Confirmation message. Returns an error if no image has been captured yet or if the session is not available.

## User Interaction

When deriving from `CameraObserver`, the user has direct access to:

- `self._cached_camera_frame` — raw PNG bytes of the last captured frame.
- `self._cached_shape` — tuple of (width, height) of the cached frame.
- `self._driver` — the `CameraDriver` instance used for hardware access.
- `self._scaling` — optional scaling factor applied during capture.

These can be used to preprocess the image (e.g., crop, filter, annotate) before the agent sees it.
