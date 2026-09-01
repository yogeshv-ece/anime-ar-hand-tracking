"""
gestures.py

Turns per-hand FingerState (+ some derived geometry) into named gesture
matches. Designed to be user-customizable: gestures are declared as data
(GestureDefinition), not hardcoded if/else chains, so you can add or tweak
gestures without touching recognition logic.

Default gestures (change these to whatever you actually want to perform):

  CHIDORI (one-handed):
      index  = extended
      middle = folded
      ring   = folded
      pinky  = folded
      thumb  = don't care
    (A "blade/knife hand with index pointing" style shape -- swap the
    pattern below for whatever single-hand shape you actually use.)

  RASENGAN (two-handed):
      Both hands roughly open (index+middle+ring+pinky extended on both
      hands), palms facing each other, within a plausible distance range
      to be "cupping" something between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.hand_geometry import FingerState, HandFrame, Handedness


FingerPattern = Tuple[Optional[bool], Optional[bool], Optional[bool], Optional[bool], Optional[bool]]
# order: (thumb, index, middle, ring, pinky). None = "don't care".


@dataclass
class GestureDefinition:
    """A declarative, user-editable single-hand gesture definition."""
    name: str
    pattern: FingerPattern

    def matches(self, fingers: FingerState) -> bool:
        actual = fingers.as_tuple()
        for wanted, got in zip(self.pattern, actual):
            if wanted is not None and wanted != got:
                return False
        return True


# ---------------------------------------------------------------------------
# Default single-hand gesture library -- EDIT THESE to customize gestures.
# ---------------------------------------------------------------------------

CHIDORI_GESTURE = GestureDefinition(
    name="chidori",
    # thumb: don't care, index: extended, middle/ring/pinky: folded
    pattern=(None, True, False, False, False),
)

# The "open hand" shape each hand must make for Rasengan charging.
RASENGAN_HAND_SHAPE = GestureDefinition(
    name="rasengan_hand_open",
    pattern=(None, True, True, True, True),
)


@dataclass
class TwoHandGeometry:
    """Derived geometry from a pair of tracked hands, used for Rasengan."""
    left_center: np.ndarray
    right_center: np.ndarray
    midpoint: np.ndarray
    distance: float
    angle_deg: float  # angle of the vector left->right, image-space degrees


def compute_two_hand_geometry(left: HandFrame, right: HandFrame) -> TwoHandGeometry:
    left_c = left.palm_center
    right_c = right.palm_center
    mid = (left_c + right_c) / 2.0
    dist = float(np.linalg.norm(right_c - left_c))
    vec = right_c - left_c
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return TwoHandGeometry(
        left_center=left_c, right_center=right_c, midpoint=mid, distance=dist,
        angle_deg=angle,
    )


def recognize_chidori(hand: HandFrame, gesture: GestureDefinition = CHIDORI_GESTURE) -> bool:
    return gesture.matches(hand.fingers)


def recognize_rasengan_shape(
    left: HandFrame,
    right: HandFrame,
    shape: GestureDefinition = RASENGAN_HAND_SHAPE,
    min_dist_ratio: float = 0.6,
    max_dist_ratio: float = 3.5,
) -> bool:
    """
    Both hands must match the required open-hand shape AND be within a
    plausible "cupping something between them" distance range, expressed
    relative to average hand scale so it works at any distance from the
    camera.
    """
    if not (shape.matches(left.fingers) and shape.matches(right.fingers)):
        return False

    avg_scale = (left.scale_ref + right.scale_ref) / 2.0
    if avg_scale <= 1e-3:
        return False

    dist = float(np.linalg.norm(left.palm_center - right.palm_center))
    ratio = dist / avg_scale

    return min_dist_ratio <= ratio <= max_dist_ratio
