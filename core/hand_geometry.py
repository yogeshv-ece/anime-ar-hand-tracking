"""
hand_geometry.py

Pure geometry helpers built on top of MediaPipe's 21 hand landmarks.
No OpenCV/rendering concerns live here -- this module only turns raw
landmark coordinates into numbers we can reason about: palm center,
orientation angle, a scale reference, and per-finger extended/folded
state.

MediaPipe hand landmark indices (for reference):

    0  WRIST
    1  THUMB_CMC        2  THUMB_MCP        3  THUMB_IP        4  THUMB_TIP
    5  INDEX_MCP        6  INDEX_PIP        7  INDEX_DIP       8  INDEX_TIP
    9  MIDDLE_MCP       10 MIDDLE_PIP       11 MIDDLE_DIP      12 MIDDLE_TIP
    13 RING_MCP         14 RING_PIP         15 RING_DIP        16 RING_TIP
    17 PINKY_MCP        18 PINKY_PIP        19 PINKY_DIP       20 PINKY_TIP
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Landmark index constants (readability > magic numbers)
# ---------------------------------------------------------------------------

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Landmarks used to define the palm polygon (excludes fingertips on purpose --
# these five points barely move when fingers curl, which is what makes them
# a stable anchor for "palm center").
PALM_LANDMARKS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


class Handedness(Enum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class FingerState:
    """Extended/folded booleans for all five fingers."""
    thumb: bool = False
    index: bool = False
    middle: bool = False
    ring: bool = False
    pinky: bool = False

    def as_tuple(self) -> Tuple[bool, bool, bool, bool, bool]:
        return (self.thumb, self.index, self.middle, self.ring, self.pinky)

    def __repr__(self) -> str:
        names = "TIMRP"
        flags = "".join(n if v else "-" for n, v in zip(names, self.as_tuple()))
        return f"FingerState({flags})"


@dataclass
class HandFrame:
    """
    A single hand's fully-derived geometry for one video frame.
    Coordinates are pixel coordinates (x, y) in the current frame;
    z is MediaPipe's relative depth (smaller = closer to camera).
    """
    landmarks_px: np.ndarray          # (21, 2) float32 pixel coords
    landmarks_z: np.ndarray           # (21,) relative depth
    handedness: Handedness
    detection_confidence: float

    palm_center: np.ndarray = field(default_factory=lambda: np.zeros(2))
    orientation_deg: float = 0.0      # 0 = fingers pointing up (-Y in image space)
    scale_ref: float = 1.0            # characteristic hand size in pixels
    fingers: FingerState = field(default_factory=FingerState)

    def __post_init__(self):
        self.palm_center = compute_palm_center(self.landmarks_px)
        self.orientation_deg = compute_hand_orientation(self.landmarks_px)
        self.scale_ref = compute_scale_reference(self.landmarks_px)
        self.fingers = compute_finger_states(self.landmarks_px, self.handedness)


# ---------------------------------------------------------------------------
# Core geometry functions
# ---------------------------------------------------------------------------

def compute_palm_center(landmarks_px: np.ndarray) -> np.ndarray:
    """
    Average of wrist + 4 MCP joints. This is far more stable than using
    a single landmark (e.g. wrist alone) because it's a centroid of five
    points spread across the palm -- noise on any one landmark is damped.
    """
    pts = landmarks_px[PALM_LANDMARKS]
    return pts.mean(axis=0)


def compute_hand_orientation(landmarks_px: np.ndarray) -> float:
    """
    Orientation angle in degrees, where 0 degrees means the hand's "up"
    axis (wrist -> middle-finger-MCP) points straight up in image space.

    We use wrist -> middle MCP rather than wrist -> middle TIP because the
    MCP joint barely moves when fingers curl/extend, giving a much more
    stable rotation reference than a fingertip would.

    Returned angle increases clockwise (standard image-space convention,
    since image Y grows downward), range (-180, 180].
    """
    wrist = landmarks_px[WRIST]
    middle_mcp = landmarks_px[MIDDLE_MCP]
    direction = middle_mcp - wrist  # (dx, dy)

    # atan2(dx, -dy): angle from "straight up" (-Y), positive = clockwise.
    angle = math.degrees(math.atan2(direction[0], -direction[1]))
    return angle


def compute_scale_reference(landmarks_px: np.ndarray) -> float:
    """
    Characteristic hand size in pixels, used to scale VFX so it stays
    visually proportional to apparent hand size regardless of distance
    from the camera.

    We use the distance from wrist to middle-finger MCP combined with
    the palm width (index MCP to pinky MCP), averaged, since either one
    alone can be distorted by hand rotation relative to the camera.
    """
    wrist = landmarks_px[WRIST]
    middle_mcp = landmarks_px[MIDDLE_MCP]
    index_mcp = landmarks_px[INDEX_MCP]
    pinky_mcp = landmarks_px[PINKY_MCP]

    palm_length = np.linalg.norm(middle_mcp - wrist)
    palm_width = np.linalg.norm(index_mcp - pinky_mcp)

    return float((palm_length + palm_width) / 2.0)


def _finger_extended(
    landmarks_px: np.ndarray,
    mcp_idx: int,
    pip_idx: int,
    dip_idx: int,
    tip_idx: int,
    palm_center: np.ndarray,
) -> bool:
    """
    A finger is "extended" if:
      1. The tip is farther from the palm center than the PIP joint is
         (i.e. the finger is stretched outward, not curled back over
         the palm), AND
      2. The joint chain MCP->PIP->DIP->TIP is reasonably straight
         (sum of segment lengths is close to the direct MCP->TIP
         distance -- a curled finger has a much shorter direct distance
         relative to its segment-length sum).

    Using both distance-from-palm AND straightness avoids the classic
    false positive where a folded finger's tip still happens to be
    slightly farther from the palm center than its PIP (common with
    fingers folded sideways rather than straight down).
    """
    mcp, pip, dip, tip = (
        landmarks_px[mcp_idx],
        landmarks_px[pip_idx],
        landmarks_px[dip_idx],
        landmarks_px[tip_idx],
    )

    tip_to_palm = np.linalg.norm(tip - palm_center)
    pip_to_palm = np.linalg.norm(pip - palm_center)
    farther_than_pip = tip_to_palm > pip_to_palm * 1.05  # small margin vs noise

    seg_sum = (
        np.linalg.norm(pip - mcp)
        + np.linalg.norm(dip - pip)
        + np.linalg.norm(tip - dip)
    )
    direct = np.linalg.norm(tip - mcp)
    straightness = direct / (seg_sum + 1e-6)  # 1.0 = perfectly straight

    is_straight = straightness > 0.8

    return bool(farther_than_pip and is_straight)


def _thumb_extended(
    landmarks_px: np.ndarray, handedness: Handedness, palm_center: np.ndarray
) -> bool:
    """
    Thumb extension can't use the same finger-flexion test because the
    thumb's joints move in a different plane. Instead we check whether
    the thumb tip is displaced sideways (away from the palm, away from
    the other fingers) beyond a threshold relative to hand scale, using
    the sign convention that depends on handedness (mirrored geometry).
    """
    thumb_tip = landmarks_px[THUMB_TIP]
    thumb_mcp = landmarks_px[THUMB_MCP]
    index_mcp = landmarks_px[INDEX_MCP]
    pinky_mcp = landmarks_px[PINKY_MCP]

    hand_width = np.linalg.norm(index_mcp - pinky_mcp) + 1e-6

    # Distance from thumb tip to the index-MCP..pinky-MCP line approximates
    # how far the thumb sticks out sideways from the palm.
    line_vec = pinky_mcp - index_mcp
    line_len = np.linalg.norm(line_vec) + 1e-6
    line_unit = line_vec / line_len
    to_tip = thumb_tip - index_mcp
    proj_len = np.dot(to_tip, line_unit)
    proj_point = index_mcp + proj_len * line_unit
    perp_dist = np.linalg.norm(thumb_tip - proj_point)

    # Also require the thumb tip to be farther from the palm center than
    # the thumb MCP, i.e. actually stretched out and not tucked in.
    tip_far_enough = np.linalg.norm(thumb_tip - palm_center) > np.linalg.norm(
        thumb_mcp - palm_center
    ) * 1.1

    return bool((perp_dist / hand_width > 0.35) and tip_far_enough)


def compute_finger_states(
    landmarks_px: np.ndarray, handedness: Handedness
) -> FingerState:
    palm_center = compute_palm_center(landmarks_px)

    return FingerState(
        thumb=_thumb_extended(landmarks_px, handedness, palm_center),
        index=_finger_extended(
            landmarks_px, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP, palm_center
        ),
        middle=_finger_extended(
            landmarks_px, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP, palm_center
        ),
        ring=_finger_extended(
            landmarks_px, RING_MCP, RING_PIP, RING_DIP, RING_TIP, palm_center
        ),
        pinky=_finger_extended(
            landmarks_px, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP, palm_center
        ),
    )


def landmarks_to_pixel_array(
    landmark_list, frame_w: int, frame_h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a flat list of 21 normalized landmarks (each with .x/.y/.z in
    [0, 1] image-relative coordinates) into:
      - (21, 2) pixel-space xy array
      - (21,) relative z array

    Works directly with the modern `mediapipe.tasks` API, where
    `HandLandmarkerResult.hand_landmarks[i]` is already such a flat list
    (unlike the legacy `mp.solutions.hands` API, which wrapped it in a
    `.landmark` attribute -- there is no `.landmark` indirection here).
    """
    xy = np.array(
        [[lm.x * frame_w, lm.y * frame_h] for lm in landmark_list],
        dtype=np.float32,
    )
    z = np.array([lm.z for lm in landmark_list], dtype=np.float32)
    return xy, z
