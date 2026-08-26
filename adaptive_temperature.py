"""Small feedback controller for rollout sampling temperature.

The controller keeps the confidence distribution close to the first healthy
training window.  If confidence rises, temperature rises; if confidence falls,
temperature falls.  A small secondary term raises temperature when both
effective answer diversity and the fraction of groups with real gradients fall.

There is deliberately no step-based schedule in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Optional


@dataclass
class AdaptiveTemperatureController:
    """Update temperature from one window of globally aggregated metrics."""

    temperature: float = 1.0
    min_temperature: float = 0.5
    max_temperature: float = 1.5
    confidence_gain: float = 0.5
    valid_group_gain: float = 0.05
    confidence_deadband: float = 0.005
    max_change: float = 0.05

    reference_confidence: Optional[float] = None
    reference_effective_diversity: Optional[float] = None
    reference_valid_group_rate: Optional[float] = None

    def __post_init__(self) -> None:
        if not 0.0 < self.min_temperature <= self.temperature:
            raise ValueError("min_temperature must be > 0 and <= temperature")
        if self.max_temperature < self.temperature:
            raise ValueError("max_temperature must be >= temperature")
        if self.confidence_gain < 0.0 or self.valid_group_gain < 0.0:
            raise ValueError("controller gains must be non-negative")
        if self.confidence_deadband < 0.0:
            raise ValueError("confidence_deadband must be non-negative")
        if not 0.0 < self.max_change < 1.0:
            raise ValueError("max_change must be between 0 and 1")

    def update(
        self,
        mean_confidence: float,
        effective_diversity: float,
        valid_group_rate: float,
    ) -> Dict[str, float | str]:
        """Calibrate on the first window, then return the next temperature.

        The update is multiplicative in log-temperature space:

            delta = confidence_gain * (confidence - reference_confidence)

        When confidence has not fallen, a loss of both diversity and valid
        gradient groups adds a small positive exploration term.
        """
        old_temperature = float(self.temperature)

        if self.reference_confidence is None:
            self.reference_confidence = float(mean_confidence)
            self.reference_effective_diversity = float(effective_diversity)
            self.reference_valid_group_rate = float(valid_group_rate)
            return {
                "phase": "calibration",
                "old_temperature": old_temperature,
                "new_temperature": old_temperature,
                "confidence_error": 0.0,
                "exploration_bonus": 0.0,
                "delta_log_temperature": 0.0,
            }

        confidence_error = mean_confidence - self.reference_confidence
        if abs(confidence_error) <= self.confidence_deadband:
            confidence_error = 0.0

        delta = self.confidence_gain * confidence_error
        exploration_bonus = 0.0

        diversity_fell = (
            effective_diversity < self.reference_effective_diversity
        )
        valid_rate_fell = valid_group_rate < self.reference_valid_group_rate
        confidence_not_low = (
            mean_confidence
            >= self.reference_confidence - self.confidence_deadband
        )

        # This term targets AR-style overconfidence.  It is disabled when
        # confidence is falling, so it cannot fight the Block Diffusion
        # controller by increasing temperature during an under-confidence
        # collapse.
        if diversity_fell and valid_rate_fell and confidence_not_low:
            exploration_bonus = self.valid_group_gain * (
                self.reference_valid_group_rate - valid_group_rate
            )
            delta += exploration_bonus

        max_log_change = math.log1p(self.max_change)
        delta = max(-max_log_change, min(max_log_change, delta))

        new_temperature = old_temperature * math.exp(delta)
        new_temperature = max(
            self.min_temperature,
            min(self.max_temperature, new_temperature),
        )
        self.temperature = float(new_temperature)

        return {
            "phase": "update",
            "old_temperature": old_temperature,
            "new_temperature": self.temperature,
            "confidence_error": float(confidence_error),
            "exploration_bonus": float(exploration_bonus),
            "delta_log_temperature": float(delta),
        }
