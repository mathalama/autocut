"""Reliable voice activity detection with adaptive energy gating."""

import numpy as np


def calculate_rms(audio: np.ndarray) -> float:
    """Calculate Root Mean Square (RMS) energy of an audio buffer."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


class VoiceActivityDetector:
    """Adaptive energy-based VAD with robust noise-floor tracking for noisy microphones."""

    def __init__(self, energy_threshold: float = 0.015) -> None:
        self.energy_threshold = energy_threshold
        self.noise_floor = 0.005
        self._history: list[float] = []

    def is_speech_active(self, audio: np.ndarray) -> bool:
        """Check whether the given buffer contains audible speech above background noise."""
        if len(audio) < 128:
            return False

        rms = calculate_rms(audio)

        # Track recent energy levels to adapt to steady background noise (fans, hum)
        self._history.append(rms)
        if len(self._history) > 50:  # ~5 seconds rolling window (50 * 100ms)
            self._history.pop(0)

        # Noise floor is estimated from the lower 25th percentile of recent audio
        if len(self._history) >= 10:
            sorted_history = sorted(self._history)
            estimated_floor = sorted_history[len(sorted_history) // 4]
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * estimated_floor

        # Speech must be clearly above both minimum sensitivity and the dynamic noise floor
        dynamic_threshold = max(self.energy_threshold, self.noise_floor * 2.2)

        return rms > dynamic_threshold
