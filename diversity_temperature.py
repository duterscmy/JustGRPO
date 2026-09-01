"""Diversity-only feedback controller for rollout temperature.

The controller has one goal: keep the recent mean number of distinct answers
close to ``target_diversity``.  It does not use rollout confidence, reward,
accuracy, training step, or a pre-defined temperature schedule.

If diversity is too low, temperature increases.  If diversity is too high,
temperature decreases.  Updates are multiplicative so the same settings work
for different initial temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional


@dataclass
class DiversityTemperatureController:
    """Track answer diversity and return the temperature for the next step."""

    temperature: float = 0.6
    target_diversity: float = 2.0
    ema_decay: float = 0.6
    gain: float = 0.5
    deadband: float = 0.1
    max_change: float = 0.10
    min_temperature: float = 0.3
    max_temperature: float = 1.5

    smoothed_diversity: Optional[float] = None
    update_count: int = 0

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")
        if self.target_diversity <= 0.0:
            raise ValueError("target_diversity must be greater than zero")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.gain < 0.0:
            raise ValueError("gain must be non-negative")
        if self.deadband < 0.0:
            raise ValueError("deadband must be non-negative")
        if not 0.0 <= self.max_change < 1.0:
            raise ValueError("max_change must be in [0, 1)")
        if not 0.0 < self.min_temperature <= self.temperature:
            raise ValueError(
                "min_temperature must be > 0 and <= initial temperature"
            )
        if self.max_temperature < self.temperature:
            raise ValueError(
                "max_temperature must be >= initial temperature"
            )

    def update(self, mean_diversity: float) -> Dict[str, float | str]:
        """Update once from the globally averaged diversity of one train step.

        ``deadband`` is measured in distinct answers.  Outside that band, the
        remaining error is normalized by the target and converted into a
        multiplicative temperature change.  ``max_change`` is the maximum
        relative change in either direction for a single optimizer step.
        """
        if not math.isfinite(mean_diversity) or mean_diversity <= 0.0:
            raise ValueError("mean_diversity must be finite and > 0")

        old_temperature = float(self.temperature)
        mean_diversity = float(mean_diversity)

        if self.smoothed_diversity is None:
            self.smoothed_diversity = mean_diversity
        else:
            self.smoothed_diversity = (
                self.ema_decay * self.smoothed_diversity
                + (1.0 - self.ema_decay) * mean_diversity
            )

        raw_error = self.target_diversity - self.smoothed_diversity
        if abs(raw_error) <= self.deadband:
            controlled_error = 0.0
        else:
            controlled_error = math.copysign(
                abs(raw_error) - self.deadband,
                raw_error,
            )

        normalized_error = controlled_error / self.target_diversity
        raw_log_change = self.gain * normalized_error
        raw_factor = math.exp(raw_log_change)
        min_factor = 1.0 - self.max_change
        max_factor = 1.0 + self.max_change
        applied_factor = min(max(raw_factor, min_factor), max_factor)

        new_temperature = old_temperature * applied_factor
        new_temperature = min(
            max(new_temperature, self.min_temperature),
            self.max_temperature,
        )
        self.temperature = float(new_temperature)
        self.update_count += 1

        if self.temperature > old_temperature + 1e-12:
            action = "increase"
        elif self.temperature < old_temperature - 1e-12:
            action = "decrease"
        else:
            action = "hold"

        return {
            "action": action,
            "old_temperature": old_temperature,
            "new_temperature": self.temperature,
            "mean_diversity": mean_diversity,
            "smoothed_diversity": float(self.smoothed_diversity),
            "raw_error": float(raw_error),
            "controlled_error": float(controlled_error),
            "normalized_error": float(normalized_error),
            "raw_log_change": float(raw_log_change),
            "applied_relative_change": (
                self.temperature / old_temperature - 1.0
            ),
        }
