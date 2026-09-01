"""
vfx_assets.py

RECOMMENDATION ON ALPHA FORMAT (read this before exporting from After Effects)
-------------------------------------------------------------------------------
OpenCV's VideoCapture (built on FFmpeg) does NOT reliably decode alpha
channels from WebM (VP9) or MOV (ProRes 4444 / PNG codec) across platforms.
Alpha support depends entirely on how your OpenCV/FFmpeg build was compiled,
and even when it works, `cv2.VideoCapture` only ever hands you a 3-channel
BGR frame -- there is no supported code path to pull a 4th alpha channel out
of `cap.read()`. People do hack around this (e.g. exporting the video as a
side-by-side "RGB + grayscale matte" composite and splitting it back apart
in code), but that's extra decode complexity and fragile across FFmpeg builds
for zero real benefit here.

RECOMMENDED APPROACH: RGBA PNG image sequences.

    - Export from After Effects as PNG sequence (Format: PNG, RGB + Alpha).
    - cv2.imread(path, cv2.IMREAD_UNCHANGED) reliably gives you a true
      (H, W, 4) BGRA array on every platform, every OpenCV build. No FFmpeg
      alpha ambiguity at all.
    - Compositing is then a straightforward alpha-blend per frame, which is
      what `composite_bgra_onto` below does.
    - Downsides vs a video container: more files on disk, slightly more
      I/O at load time. Irrelevant here because we load every frame once
      into memory at startup and play back from RAM -- see VFXClip.

    If you strongly prefer a single-file asset, RGBA-supporting WebM/MOV can
    still be *dropped in* later by swapping VFXClip's loader for a
    frame-by-frame FFmpeg pipe (e.g. `ffmpeg -i in.mov -pix_fmt bgra -f
    rawvideo -`) that reads raw BGRA bytes directly, bypassing cv2.VideoCapture
    entirely. That's the correct escape hatch if you outgrow PNG sequences --
    not cv2.VideoCapture, which cannot do it.

FOLDER CONVENTION
------------------
Each VFX clip is a folder of numbered PNG frames:

    assets/
        chidori/
            chidori_burst/
                0001.png
                0002.png
                ...
            chidori_loop/
                0001.png
                ...
        rasengan/
            rasengan_burst/
                0001.png
            rasengan_loop/
                0001.png

If you only have .webm/.mov files today, see `scripts/extract_png_sequence.py`
(a tiny ffmpeg wrapper) to convert them once, offline, at setup time -- ffmpeg
CAN decode alpha from those containers correctly on the command line even
though cv2.VideoCapture in Python can't surface it; we just use ffmpeg as an
offline conversion step rather than a live-decode dependency.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class VFXClip:
    """
    A loaded, ready-to-play sequence of BGRA frames held fully in memory.
    Loading every frame up front means playback is just array indexing --
    no per-frame disk I/O or decode cost inside the render loop.
    """
    name: str
    frames: List[np.ndarray] = field(default_factory=list)  # each (h, w, 4) BGRA
    fps: float = 30.0
    loop: bool = False

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def is_empty(self) -> bool:
        return self.frame_count == 0

    @classmethod
    def load_from_folder(cls, folder: str, name: Optional[str] = None,
                          fps: float = 30.0, loop: bool = False) -> "VFXClip":
        name = name or os.path.basename(os.path.normpath(folder))
        paths = sorted(glob.glob(os.path.join(folder, "*.png")))
        frames: List[np.ndarray] = []

        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                # grayscale, no alpha -- treat as fully opaque
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                # no alpha channel present -- treat as fully opaque
                b, g, r = cv2.split(img)
                alpha = np.full_like(b, 255)
                img = cv2.merge([b, g, r, alpha])
            frames.append(img)

        if not frames:
            print(f"[vfx_assets] WARNING: no PNG frames found in '{folder}'. "
                  f"Using a placeholder magenta clip so the app keeps running.")
            frames = [_placeholder_frame(name)]

        return cls(name=name, frames=frames, fps=fps, loop=loop)


def _placeholder_frame(label: str, size: int = 256) -> np.ndarray:
    """A visible placeholder so a missing asset fails loudly, not silently."""
    frame = np.zeros((size, size, 4), dtype=np.uint8)
    cv2.circle(frame, (size // 2, size // 2), size // 2 - 10, (255, 0, 255, 200), 6)
    cv2.putText(frame, "MISSING", (size // 2 - 90, size // 2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    cv2.putText(frame, label[:14], (size // 2 - 90, size // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255, 255), 2)
    return frame


class VFXPlayer:
    """
    Plays a sequence of clips as a single logical "effect" (burst -> loop),
    treating playback as *state*, not a per-frame event:

      - play_burst_then_loop(): starts the burst if not already mid-burst;
        once the burst finishes, automatically transitions into the loop
        and stays there until told to stop.
      - stop(): begins a smooth fade-out from the current frame rather
        than an instant cut.
      - The burst is never restarted just because this method is called
        again on a later frame while already active -- callers can safely
        call it every frame from the render loop.
    """

    def __init__(self, burst: VFXClip, loop: VFXClip, fade_frames: int = 8):
        self.burst = burst
        self.loop = loop
        self.fade_frames = max(1, fade_frames)

        self._playing = False
        self._in_burst = False
        self._frame_idx = 0
        self._fade_alpha = 0.0     # 0..1, current overall opacity multiplier
        self._fading_out = False

    def start(self) -> None:
        if self._playing and not self._fading_out:
            return  # already active mid-burst-or-loop: do not restart burst
        self._playing = True
        self._fading_out = False
        self._in_burst = True
        self._frame_idx = 0
        # keep existing fade_alpha if we were mid-fade-out (resume smoothly)

    def stop(self) -> None:
        if self._playing:
            self._fading_out = True

    @property
    def active(self) -> bool:
        return self._playing

    def get_frame(self) -> Optional[np.ndarray]:
        """Advance one frame of playback and return the current BGRA frame
        (with fade applied to its alpha channel), or None if fully stopped."""
        if not self._playing:
            return None

        clip = self.burst if self._in_burst else self.loop
        if clip.is_empty:
            return None

        frame = clip.frames[min(self._frame_idx, clip.frame_count - 1)].copy()

        # advance fade
        if self._fading_out:
            self._fade_alpha = max(0.0, self._fade_alpha - 1.0 / self.fade_frames)
        else:
            self._fade_alpha = min(1.0, self._fade_alpha + 1.0 / self.fade_frames)

        if self._fade_alpha <= 0.0 and self._fading_out:
            self._playing = False
            return None

        if self._fade_alpha < 1.0:
            frame = frame.copy()
            frame[:, :, 3] = (frame[:, :, 3].astype(np.float32) * self._fade_alpha).astype(np.uint8)

        # advance playback position
        self._frame_idx += 1
        if self._in_burst and self._frame_idx >= clip.frame_count:
            self._in_burst = False
            self._frame_idx = 0
        elif not self._in_burst and self._frame_idx >= clip.frame_count:
            self._frame_idx = 0  # loop clip wraps

        return frame


@dataclass
class VFXLibrary:
    """Holds all loaded clips for both effects."""
    chidori_burst: VFXClip
    chidori_loop: VFXClip
    rasengan_burst: VFXClip
    rasengan_loop: VFXClip

    @classmethod
    def load(cls, assets_root: str) -> "VFXLibrary":
        def _load(effect: str, clip: str, loop: bool) -> VFXClip:
            folder = os.path.join(assets_root, effect, clip)
            return VFXClip.load_from_folder(folder, name=clip, loop=loop)

        return cls(
            chidori_burst=_load("chidori", "chidori_burst", loop=False),
            chidori_loop=_load("chidori", "chidori_loop", loop=True),
            rasengan_burst=_load("rasengan", "rasengan_burst", loop=False),
            rasengan_loop=_load("rasengan", "rasengan_loop", loop=True),
        )


def composite_bgra_onto(
    background_bgr: np.ndarray,
    overlay_bgra: np.ndarray,
    center_xy: tuple,
    scale: float = 1.0,
    rotation_deg: float = 0.0,
) -> np.ndarray:
    """
    Alpha-composite `overlay_bgra` onto `background_bgr` at `center_xy`,
    with the overlay scaled and rotated first via an affine warp.

    Returns a new BGR frame (background_bgr is not modified in place).
    """
    if overlay_bgra is None or overlay_bgra.size == 0:
        return background_bgr

    oh, ow = overlay_bgra.shape[:2]
    if oh == 0 or ow == 0 or scale <= 0:
        return background_bgr

    # Rotate + scale about the overlay's own center, then translate so that
    # center lands on center_xy in the background frame.
    M = cv2.getRotationMatrix2D((ow / 2.0, oh / 2.0), rotation_deg, scale)
    # After rotation+scale the overlay may extend beyond its original canvas;
    # warp into a canvas sized to the background so we don't clip during warp.
    bh, bw = background_bgr.shape[:2]
    M[0, 2] += center_xy[0] - ow / 2.0
    M[1, 2] += center_xy[1] - oh / 2.0

    warped = cv2.warpAffine(
        overlay_bgra, M, (bw, bh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    overlay_rgb = warped[:, :, :3].astype(np.float32)
    alpha = (warped[:, :, 3:4].astype(np.float32)) / 255.0

    bg = background_bgr.astype(np.float32)
    out = overlay_rgb * alpha + bg * (1.0 - alpha)
    return out.astype(np.uint8)
