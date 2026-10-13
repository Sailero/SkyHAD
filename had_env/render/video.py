"""Optional streaming video capture from the standard ``env.render()`` API."""
from __future__ import annotations

from importlib import import_module
import math
from numbers import Real
from pathlib import Path

import numpy as np


class VideoRecorder:
    """Write one rendered RGB frame per ``capture`` without stepping the env.

    The caller owns reset, actions, timing and environment lifetime. Frames are
    streamed to FFmpeg instead of retained in memory. Install ``had-env[record]``
    for encoding; importing this module does not import the optional encoder.
    Odd image dimensions receive one repeated edge pixel on the right/bottom
    for H.264 compatibility. Original pixels and pixel aspect stay unchanged.

    Example::

        with VideoRecorder(env, "episode.mp4") as video:
            video.capture()
            while env.agents:
                env.step({a: env.action_space(a).sample() for a in env.agents})
                video.capture()
    """

    def __init__(self, env, path, fps=None):
        if getattr(env, "render_mode", None) != "rgb_array":
            raise ValueError("VideoRecorder requires an environment with render_mode='rgb_array'")
        if fps is None:
            fps = (getattr(env, "metadata", None) or {}).get("render_fps", 30)
        if isinstance(fps, bool) or not isinstance(fps, Real) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be a finite positive number")
        self.env = env
        self.path = Path(path)
        self.fps = float(fps)
        self._writer = None
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("VideoRecorder is closed; create a new recorder for another stream")
        if self._writer is not None:
            raise RuntimeError("VideoRecorder context is already open")
        try:
            imageio = import_module("imageio.v2")
            import_module("imageio_ffmpeg")
        except ImportError as exc:
            raise RuntimeError('Video recording requires imageio and FFmpeg; install with pip install -e ".[record]"') from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(str(self.path), format="FFMPEG", fps=self.fps,
                                          codec="libx264", macro_block_size=1)
        return self

    def capture(self):
        """Render and immediately encode the current frame; never advance physics."""
        if self._closed:
            raise RuntimeError("VideoRecorder is closed")
        if self._writer is None:
            raise RuntimeError("Enter the VideoRecorder context before capturing frames")
        if getattr(self.env, "render_mode", None) != "rgb_array":
            raise ValueError("VideoRecorder requires render_mode='rgb_array'")
        frame = self.env.render()
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]):
            raise ValueError("env.render() must return a nonempty RGB uint8 array with shape (height, width, 3)")
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            frame = np.pad(frame, ((0, frame.shape[0] % 2), (0, frame.shape[1] % 2), (0, 0)), mode="edge")
        self._writer.append_data(frame)

    def close(self):
        """Finalize the encoder once, without closing the independently owned env."""
        if self._closed:
            return
        self._closed = True
        writer, self._writer = self._writer, None
        if writer is not None:
            writer.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
