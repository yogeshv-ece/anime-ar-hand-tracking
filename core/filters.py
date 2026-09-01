"""
filters.py

Smoothing/filtering utilities.

CHOICE MADE: One Euro Filter, with a plain EMA fallback for angle-wrap-safe
quantities where needed.

Why One Euro over plain EMA or a Kalman filter:

- Plain EMA (single fixed alpha) forces a trade-off you can't escape: a
  low alpha kills jitter but adds noticeable lag when the hand moves fast
  (e.g. a quick Rasengan charge-up motion); a high alpha tracks fast
  motion but lets landmark jitter straight through when the hand is
  nearly still (e.g. holding a Chidori pose). Hand tracking needs both
  regimes in the same session.

- The One Euro Filter (Casiez et al., 2012) is specifically designed for
  this: it adapts its cutoff frequency based on the estimated signal
  speed. Slow/still signal -> aggressive smoothing (low cutoff). Fast
  signal -> lighter smoothing (high cutoff), so responsiveness isn't
  sacrificed during quick gestures. It's the standard choice in
  interactive/AR pointer-smoothing (it was literally designed for noisy
  interactive input like this).

- A Kalman filter would be justified if we wanted a proper motion MODEL
  (e.g. predicting through a full occlusion using constant-velocity
  assumptions) or needed to fuse multiple noisy sensors. For this
  project we don't have a reliable process model for arbitrary hand
  motion and we don't need multi-sensor fusion, so a Kalman filter would
  add tuning complexity (process/measurement covariance matrices) without
  a corresponding benefit over One Euro. Not using it here; noted as a
  reasonable v2 upgrade if we later want predictive tracking through
  brief occlusions.

This module provides:
    - OneEuroFilter: filters a single scalar
    - OneEuroVectorFilter: filters an (N,) or (N,2) array, one Euro filter
      per component
    - AngleEMA: an EMA specialized for angles that handles wraparound
      (e.g. going from 179 degrees to -179 degrees shouldn't be treated
      as a 358-degree jump)
"""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np


class _LowPassFilter:
    def __init__(self):
        self._initialized = False
        self._value = 0.0

    def filter(self, value: float, alpha: float) -> float:
        if not self._initialized:
            self._value = value
            self._initialized = True
        else:
            self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value

    @property
    def value(self) -> float:
        return self._value


class OneEuroFilter:
    """
    One Euro Filter for a single scalar signal.

    Parameters
    ----------
    min_cutoff : float
        Minimum cutoff frequency. Lower = more smoothing when signal is
        slow/still, but more lag on sudden changes.
    beta : float
        Speed coefficient. Higher = cutoff increases more aggressively
        with signal speed, i.e. faster motion is smoothed less.
    d_cutoff : float
        Cutoff frequency used to filter the derivative estimate itself.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff

        self._x_filter = _LowPassFilter()
        self._dx_filter = _LowPassFilter()
        self._last_time: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value: float, timestamp: Optional[float] = None) -> float:
        now = timestamp if timestamp is not None else time.monotonic()

        if self._last_time is None:
            dt = 1.0 / 30.0  # assume 30fps on first sample
        else:
            dt = max(now - self._last_time, 1e-6)
        self._last_time = now

        prev_x = self._x_filter.value if self._x_filter._initialized else value
        dx = (value - prev_x) / dt

        d_alpha = self._alpha(self.d_cutoff, dt)
        edx = self._dx_filter.filter(dx, d_alpha)

        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        return self._x_filter.filter(value, alpha)


class OneEuroVectorFilter:
    """Applies an independent OneEuroFilter to each component of an array."""

    def __init__(self, n_components: int, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0):
        self._filters = [
            OneEuroFilter(min_cutoff, beta, d_cutoff) for _ in range(n_components)
        ]

    def filter(self, values: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        out = np.array(
            [f.filter(v, timestamp) for f, v in zip(self._filters, flat)],
            dtype=np.float64,
        )
        return out.reshape(np.asarray(values).shape)


class AngleEMA:
    """
    EMA smoothing for an angle in degrees, wraparound-safe by smoothing
    the (cos, sin) components instead of the raw angle.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._cos_f = _LowPassFilter()
        self._sin_f = _LowPassFilter()

    def filter(self, angle_deg: float) -> float:
        rad = math.radians(angle_deg)
        c = self._cos_f.filter(math.cos(rad), self.alpha)
        s = self._sin_f.filter(math.sin(rad), self.alpha)
        return math.degrees(math.atan2(s, c))


class ScalarEMA:
    """Plain exponential moving average for a scalar (e.g. scale reference)."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha
        self._f = _LowPassFilter()

    def filter(self, value: float) -> float:
        return self._f.filter(value, self.alpha)
