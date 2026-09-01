"""
main.py

Entry point. Wires together:

    WEBCAM -> VIDEO CAPTURE -> HAND DETECTION -> 21 LANDMARKS ->
    GESTURE RECOGNITION -> STATE MACHINE -> POSITION/ROTATION/SCALE ->
    VFX OVERLAY -> FINAL VIDEO

MEDIAPIPE API NOTE
------------------
This uses the modern `mediapipe.tasks` HandLandmarker API, not the older
`mp.solutions.hands` API you'll see in a lot of older tutorials -- recent
mediapipe pip builds (0.10.x) no longer expose `mp.solutions` at all, so
`mp.solutions.hands.Hands(...)` will raise `AttributeError` on a fresh
install. The Tasks API needs a one-time model file download:

    curl -L -o hand_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place it next to this script (or set HAND_LANDMARKER_MODEL_PATH below).

Run:
    python main.py

Press 'q' to quit, 'd' to toggle debug landmark overlay.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

from core.hand_geometry import HandFrame, Handedness, landmarks_to_pixel_array
from core.gestures import (
    recognize_chidori,
    recognize_rasengan_shape,
    compute_two_hand_geometry,
)
from core.state_machine import GestureStateMachine, AppState
from core.filters import OneEuroVectorFilter, AngleEMA, ScalarEMA
from core.vfx_assets import VFXLibrary, VFXPlayer, composite_bgra_onto

ASSETS_ROOT = os.path.join(os.path.dirname(__file__), "assets")
HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "hand_landmarker.task"
)

# Tune these to taste -- see filters.py docstring for what they mean.
POS_FILTER_MIN_CUTOFF = 1.2
POS_FILTER_BETA = 0.4
SCALE_EMA_ALPHA = 0.2
ORIENTATION_EMA_ALPHA = 0.3

# Reference hand scale (px) at which VFX renders at 1.0x -- tune to your
# asset's native resolution / desired real-world size.
CHIDORI_REFERENCE_SCALE = 140.0
RASENGAN_REFERENCE_DIST = 220.0
MIN_VFX_SCALE = 0.2
MAX_VFX_SCALE = 1.5


class HandTrackTarget:
    """Per-hand smoothing state (position + orientation + scale)."""

    def __init__(self):
        self.pos_filter = OneEuroVectorFilter(2, min_cutoff=POS_FILTER_MIN_CUTOFF,
                                               beta=POS_FILTER_BETA)
        self.angle_ema = AngleEMA(alpha=ORIENTATION_EMA_ALPHA)
        self.scale_ema = ScalarEMA(alpha=SCALE_EMA_ALPHA)

    def update(self, palm_center: np.ndarray, orientation_deg: float, scale_ref: float,
               now: float) -> Tuple[np.ndarray, float, float]:
        pos = self.pos_filter.filter(palm_center, now)
        angle = self.angle_ema.filter(orientation_deg)
        scale = self.scale_ema.filter(scale_ref)
        return pos, angle, scale


def build_hand_frame(landmark_list, handedness_label: str, confidence: float,
                      frame_w: int, frame_h: int) -> HandFrame:
    xy, z = landmarks_to_pixel_array(landmark_list, frame_w, frame_h)
    handedness = Handedness.LEFT if handedness_label == "Left" else Handedness.RIGHT
    return HandFrame(
        landmarks_px=xy, landmarks_z=z, handedness=handedness,
        detection_confidence=confidence,
    )


# 21-landmark hand skeleton connectivity, since mp.solutions.drawing_utils
# (which used to draw this for you) is not available in this mediapipe
# build. Same connection set as MediaPipe's HAND_CONNECTIONS.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm
]


def draw_debug_landmarks(frame: np.ndarray, hand: HandFrame) -> None:
    pts = hand.landmarks_px.astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (0, 200, 0), 1)
    for x, y in pts:
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    cx, cy = int(hand.palm_center[0]), int(hand.palm_center[1])
    cv2.circle(frame, (cx, cy), 6, (0, 128, 255), -1)
    cv2.putText(frame, f"{hand.handedness.value} {hand.fingers}",
                (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)


def main() -> None:
    if not os.path.exists(HAND_LANDMARKER_MODEL_PATH):
        raise FileNotFoundError(
            f"Hand landmarker model not found at {HAND_LANDMARKER_MODEL_PATH}.\n"
            f"Download it once with:\n\n"
            f"  curl -L -o {HAND_LANDMARKER_MODEL_PATH} "
            f"https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            f"hand_landmarker/float16/1/hand_landmarker.task\n"
        )

    vfx = VFXLibrary.load(ASSETS_ROOT)
    chidori_player = VFXPlayer(vfx.chidori_burst, vfx.chidori_loop)
    rasengan_player = VFXPlayer(vfx.rasengan_burst, vfx.rasengan_loop)

    state_machine = GestureStateMachine()
    single_hand_track: Dict[str, HandTrackTarget] = {}   # keyed by "Left"/"Right"
    rasengan_mid_track = HandTrackTarget()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Check camera permissions/index.")

    debug_overlay = False

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        prev_time = time.monotonic()
        start_time = time.monotonic()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror for natural AR interaction
            h, w = frame.shape[:2]
            now = time.monotonic()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((now - start_time) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            hand_frames: List[HandFrame] = []

            for landmark_list, handedness_cats in zip(
                result.hand_landmarks, result.handedness
            ):
                label = handedness_cats[0].category_name  # "Left" / "Right"
                conf = handedness_cats[0].score
                hf = build_hand_frame(landmark_list, label, conf, w, h)
                hand_frames.append(hf)

            left_hand = next((h_ for h_ in hand_frames if h_.handedness == Handedness.LEFT), None)
            right_hand = next((h_ for h_ in hand_frames if h_.handedness == Handedness.RIGHT), None)

            chidori_raw = False
            rasengan_raw = False
            raw_angle_for_tracker: Optional[float] = None
            any_hand_tracked = len(hand_frames) > 0

            if left_hand and right_hand:
                rasengan_raw = recognize_rasengan_shape(left_hand, right_hand)
                geom = compute_two_hand_geometry(left_hand, right_hand)
                raw_angle_for_tracker = geom.angle_deg
            elif len(hand_frames) == 1:
                chidori_raw = recognize_chidori(hand_frames[0])

            state = state_machine.update(
                chidori_raw=chidori_raw,
                rasengan_raw=rasengan_raw,
                any_hand_tracked=any_hand_tracked,
                raw_angle_deg=raw_angle_for_tracker,
            )

            # ---- Chidori rendering ----
            if state == AppState.CHIDORI_ACTIVE and hand_frames:
                active_hand = hand_frames[0]
                key = active_hand.handedness.value
                tracker = single_hand_track.setdefault(key, HandTrackTarget())
                pos, angle, scale_ref = tracker.update(
                    active_hand.palm_center, active_hand.orientation_deg,
                    active_hand.scale_ref, now,
                )
                chidori_player.start()
                vfx_scale = float(np.clip(scale_ref / CHIDORI_REFERENCE_SCALE,
                                           MIN_VFX_SCALE, MAX_VFX_SCALE))
                overlay_frame = chidori_player.get_frame()
                if overlay_frame is not None:
                    frame = composite_bgra_onto(
                        frame, overlay_frame, tuple(pos), scale=0.9,
                        rotation_deg=-angle,  # cv2 rotation is counter-clockwise-positive
                    )
            else:
                chidori_player.stop()
                leftover = chidori_player.get_frame()
                if leftover is not None and hand_frames:
                    last_hand = hand_frames[0]
                    frame = composite_bgra_onto(
                        frame, leftover, tuple(last_hand.palm_center), scale=1.0,
                        rotation_deg=-last_hand.orientation_deg,
                    )

            # ---- Rasengan rendering ----
            if state in (AppState.RASENGAN_CHARGING, AppState.RASENGAN_ACTIVE) and left_hand and right_hand:
                geom = compute_two_hand_geometry(left_hand, right_hand)
                pos, _angle, _scale = rasengan_mid_track.update(
                    geom.midpoint, geom.angle_deg,
                    (left_hand.scale_ref + right_hand.scale_ref) / 2.0, now,
                )

                energy = state_machine.rotation_tracker.energy
                spin_deg_per_frame = np.clip(energy, 0, 720) / 30.0  # ~ deg per frame @30fps
                rasengan_player._spin = getattr(rasengan_player, "_spin", 0.0) + spin_deg_per_frame

                avg_scale_ref = (left_hand.scale_ref + right_hand.scale_ref) / 2.0
                vfx_scale = float(np.clip(geom.distance / RASENGAN_REFERENCE_DIST,
                                           MIN_VFX_SCALE, MAX_VFX_SCALE))

                rasengan_player.start()
                overlay_frame = rasengan_player.get_frame()
                if overlay_frame is not None:
                    frame = composite_bgra_onto(
                        frame, overlay_frame, tuple(pos), scale=0.3,
                        rotation_deg=rasengan_player._spin,
                    )
            else:
                rasengan_player.stop()
                leftover = rasengan_player.get_frame()
                if leftover is not None:
                    # Fade out in place at the last tracked midpoint rather
                    # than re-filtering a new sample (which would corrupt the
                    # OneEuroFilter's velocity estimate with a fake "hold
                    # still" sample).
                    last_pos = rasengan_mid_track.pos_filter._filters[0]._x_filter.value, \
                        rasengan_mid_track.pos_filter._filters[1]._x_filter.value
                    frame = composite_bgra_onto(
                        frame, leftover, last_pos,
                        scale=1.0, rotation_deg=getattr(rasengan_player, "_spin", 0.0),
                    )

            if debug_overlay:
                for hf in hand_frames:
                    draw_debug_landmarks(frame, hf)

            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"{state.name}  {fps:4.1f} fps", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Anime AR", frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord("q"):
                break
            elif key_pressed == ord("d"):
                debug_overlay = not debug_overlay

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
