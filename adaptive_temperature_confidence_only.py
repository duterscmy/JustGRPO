"""Confidence-only feedback controller for rollout sampling temperature.

This v2 controller deliberately keeps answer diversity and valid-group rate out
of the temperature update.  They remain useful diagnostics, but they are not
monotonic quality signals for Block Diffusion: both can fall before collapse
and diversity can rise sharply after collapse.

The controller therefore does only three things:

1. keep the initial temperature fixed for a short calibration period;
2. smooth mean rollout confidence with an exponential moving average (EMA);
3. raise or lower log-temperature according to the smoothed confidence error.

There is no step-based temperature schedule and no rollout-type-specific rule.
Rising AR confidence naturally raises temperature, while falling Block
Diffusion confidence naturally lowers it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import median
from typing import Dict, Optional


@dataclass
class ConfidenceOnlyTemperatureController:
    """Update temperature from globally aggregated rollout confidence."""

    temperature: float = 1.0
    min_temperature: float = 0.5
    max_temperature: float = 1.5
    confidence_gain: float = 1.0
    confidence_deadband: float = 0.002
    max_change: float = 0.01
    ema_decay: float = 0.8
    calibration_windows: int = 3

    reference_confidence: Optional[float] = None
    smoothed_confidence: Optional[float] = None
    calibration_values: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 < self.min_temperature <= self.temperature:
            raise ValueError("min_temperature must be > 0 and <= temperature")
        if self.max_temperature < self.temperature:
            raise ValueError("max_temperature must be >= temperature")
        if self.confidence_gain < 0.0:
            raise ValueError("confidence_gain must be non-negative")
        if self.confidence_deadband < 0.0:
            raise ValueError("confidence_deadband must be non-negative")
        if not 0.0 < self.max_change < 1.0:
            raise ValueError("max_change must be between 0 and 1")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.calibration_windows <= 0:
            raise ValueError("calibration_windows must be greater than zero")

    @property
    def calibration_progress(self) -> int:
        return min(len(self.calibration_values), self.calibration_windows)

    def update(self, mean_confidence: float) -> Dict[str, float | int | str]:
        """Calibrate first, then return the temperature for the next window."""
        if not math.isfinite(mean_confidence):
            raise ValueError("mean_confidence must be finite")
        if not 0.0 <= mean_confidence <= 1.0:
            raise ValueError("mean_confidence must be in [0, 1]")

        old_temperature = float(self.temperature)

        if len(self.calibration_values) < self.calibration_windows:
            self.calibration_values.append(float(mean_confidence))
            running_reference = float(median(self.calibration_values))
            self.reference_confidence = running_reference
            self.smoothed_confidence = running_reference

            complete = len(self.calibration_values) == self.calibration_windows
            return {
                "phase": "calibration_complete" if complete else "calibration",
                "old_temperature": old_temperature,
                "new_temperature": old_temperature,
                "mean_confidence": float(mean_confidence),
                "smoothed_confidence": running_reference,
                "reference_confidence": running_reference,
                "raw_confidence_error": 0.0,
                "confidence_error": 0.0,
                "exploration_bonus": 0.0,
                "delta_log_temperature": 0.0,
                "calibration_progress": self.calibration_progress,
            }

        assert self.reference_confidence is not None
        assert self.smoothed_confidence is not None

        self.smoothed_confidence = (
            self.ema_decay * self.smoothed_confidence
            + (1.0 - self.ema_decay) * float(mean_confidence)
        )
        raw_error = self.smoothed_confidence - self.reference_confidence
        confidence_error = (
            0.0
            if abs(raw_error) <= self.confidence_deadband
            else raw_error
        )

        delta = self.confidence_gain * confidence_error
        max_log_change = math.log1p(self.max_change)
        delta = max(-max_log_change, min(max_log_change, delta))

        new_temperature = old_temperature * math.exp(delta)
        new_temperature = max(
            self.min_temperature,
            min(self.max_temperature, new_temperature),
        )
        self.temperature = float(new_temperature)

        return {
            "phase": "hold" if confidence_error == 0.0 else "update",
            "old_temperature": old_temperature,
            "new_temperature": self.temperature,
            "mean_confidence": float(mean_confidence),
            "smoothed_confidence": float(self.smoothed_confidence),
            "reference_confidence": float(self.reference_confidence),
            "raw_confidence_error": float(raw_error),
            "confidence_error": float(confidence_error),
            "exploration_bonus": 0.0,
            "delta_log_temperature": float(delta),
            "calibration_progress": self.calibration_progress,
        }
