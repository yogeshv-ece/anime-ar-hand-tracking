"""
state_machine.py

Debounced state machine for both effects. Nothing here reacts to a single
frame's raw detection -- everything requires N consecutive frames to enter
a state and M consecutive frames of absence to leave it, which is what
prevents flicker when MediaPipe momentarily misses a landmark or a gesture
briefly drops out of the matching pattern.

States:
    IDLE
    CHIDORI_DETECTED   (debouncing on)
    CHIDORI_ACTIVE
    RASENGAN_DETECTED  (debouncing on)
    RASENGAN_CHARGING  (burst playing)
    RASENGAN_ACTIVE    (loop playing)
    LOST_TRACK

Also owns the Rasengan angular-velocity tracking: converts the raw
angle-between-hands time series into a smoothed angular velocity that
drives energy/rotation speed in the VFX layer, with a "settle to rest"
behavior when the user stops rotating their hands.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

from core.filters import AngleEMA, ScalarEMA


class AppState(Enum):
    IDLE = auto()
    CHIDORI_DETECTED = auto()
    CHIDORI_ACTIVE = auto()
    RASENGAN_DETECTED = auto()
    RASENGAN_CHARGING = auto()
    RASENGAN_ACTIVE = auto()
    LOST_TRACK = auto()


@dataclass
class DebounceCounter:
    """Generic N-frames-on / M-frames-off debouncer."""
    on_frames_required: int = 5
    off_frames_required: int = 8

    _on_streak: int = 0
    _off_streak: int = 0
    _active: bool = False

    def update(self, condition_true: bool) -> bool:
        """Feed one frame's raw boolean condition, get back the debounced state."""
        if condition_true:
            self._on_streak += 1
            self._off_streak = 0
        else:
            self._off_streak += 1
            self._on_streak = 0

        if not self._active and self._on_streak >= self.on_frames_required:
            self._active = True
        elif self._active and self._off_streak >= self.off_frames_required:
            self._active = False

        return self._active

    @property
    def is_active(self) -> bool:
        return self._active


@dataclass
class RasenganRotationTracker:
    """
    Converts raw hand-pair angle samples into a smoothed angular velocity
    (deg/sec) plus a decaying "energy" scalar used to drive VFX playback
    speed.

    - angular_velocity: signed deg/sec, smoothed via EMA so tracking noise
      doesn't cause sudden spikes.
    - energy: a non-negative scalar in roughly [0, energy_max] that rises
      toward |angular_velocity| when rotating and decays exponentially
      toward 0 when the user stops, giving a "gradually reduce" feel
      rather than an abrupt stop.
    """
    velocity_smoothing_alpha: float = 0.35
    energy_decay_per_sec: float = 4.0     # exponential decay rate when idle
    energy_attack_alpha: float = 0.5      # how fast energy rises toward target
    energy_max: float = 720.0             # deg/sec considered "full energy"

    _angle_ema: AngleEMA = field(default_factory=lambda: AngleEMA(alpha=0.5))
    _velocity_ema: ScalarEMA = field(default_factory=lambda: ScalarEMA(alpha=0.35))
    _last_angle: Optional[float] = None
    _last_time: Optional[float] = None
    energy: float = 0.0
    angular_velocity: float = 0.0

    def update(self, raw_angle_deg: float, now: Optional[float] = None) -> float:
        now = now if now is not None else time.monotonic()
        smoothed_angle = self._angle_ema.filter(raw_angle_deg)

        if self._last_angle is None or self._last_time is None:
            dt = 1.0 / 30.0
            delta = 0.0
        else:
            dt = max(now - self._last_time, 1e-6)
            delta = _shortest_angle_diff(self._last_angle, smoothed_angle)

        self._last_angle = smoothed_angle
        self._last_time = now

        raw_velocity = delta / dt
        self.angular_velocity = self._velocity_ema.filter(raw_velocity)

        target_energy = min(abs(self.angular_velocity), self.energy_max)
        if target_energy > self.energy:
            self.energy += (target_energy - self.energy) * self.energy_attack_alpha
        else:
            self.energy *= max(0.0, 1.0 - self.energy_decay_per_sec * dt)

        return self.energy

    def reset(self) -> None:
        self._last_angle = None
        self._last_time = None
        self.energy = 0.0
        self.angular_velocity = 0.0


def _shortest_angle_diff(a_from: float, a_to: float) -> float:
    """Signed shortest difference between two angles in degrees, in (-180, 180]."""
    diff = (a_to - a_from + 180.0) % 360.0 - 180.0
    return diff


@dataclass
class GestureStateMachine:
    """
    Top-level state machine. Call `update()` once per frame with raw
    (pre-debounce) booleans for "chidori gesture seen this frame" and
    "rasengan gesture seen this frame", plus whether any/both hands are
    currently tracked. Returns the resulting AppState.
    """
    chidori_debounce: DebounceCounter = field(
        default_factory=lambda: DebounceCounter(on_frames_required=5, off_frames_required=8)
    )
    rasengan_debounce: DebounceCounter = field(
        default_factory=lambda: DebounceCounter(on_frames_required=6, off_frames_required=10)
    )
    rotation_tracker: RasenganRotationTracker = field(
        default_factory=RasenganRotationTracker
    )

    state: AppState = AppState.IDLE

    # frames since Rasengan loop/charging began, used to time the burst->loop switch
    _rasengan_active_frames: int = 0
    _rasengan_burst_frames_needed: int = 15  # ~0.5s at 30fps; tune to your burst clip length

    _no_hand_streak: int = 0
    _lost_track_threshold: int = 20

    def update(
        self,
        chidori_raw: bool,
        rasengan_raw: bool,
        any_hand_tracked: bool,
        raw_angle_deg: Optional[float] = None,
    ) -> AppState:
        if not any_hand_tracked:
            self._no_hand_streak += 1
        else:
            self._no_hand_streak = 0

        chidori_on = self.chidori_debounce.update(chidori_raw)
        rasengan_on = self.rasengan_debounce.update(rasengan_raw)

        if raw_angle_deg is not None:
            self.rotation_tracker.update(raw_angle_deg)

        if self._no_hand_streak >= self._lost_track_threshold:
            self.state = AppState.LOST_TRACK
            self._rasengan_active_frames = 0
            self.rotation_tracker.reset()
            return self.state

        # Rasengan takes priority when both are somehow detected, since it
        # requires the stricter two-hand condition.
        if rasengan_on:
            if self.state in (AppState.RASENGAN_CHARGING, AppState.RASENGAN_ACTIVE):
                self._rasengan_active_frames += 1
                if self._rasengan_active_frames >= self._rasengan_burst_frames_needed:
                    self.state = AppState.RASENGAN_ACTIVE
                else:
                    self.state = AppState.RASENGAN_CHARGING
            else:
                self.state = AppState.RASENGAN_CHARGING
                self._rasengan_active_frames = 0
        elif chidori_on:
            self.state = AppState.CHIDORI_ACTIVE
            self._rasengan_active_frames = 0
            self.rotation_tracker.reset()
        else:
            self.state = AppState.IDLE
            self._rasengan_active_frames = 0
            self.rotation_tracker.reset()

        return self.state
