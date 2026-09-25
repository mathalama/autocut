"""Resilient microphone audio capture using sounddevice and circular buffer."""

import logging
import threading
import time
from typing import Any, Callable, Optional
import numpy as np
import sounddevice as sd

from autocut.config import BLOCK_SIZE, CHANNELS, SAMPLE_RATE

logger = logging.getLogger(__name__)


class MicrophoneAccessError(RuntimeError):
    """Raised when the audio recording interface cannot open the microphone."""
    pass


def list_input_devices() -> list[dict[str, Any]]:
    """Return a list of available audio input devices."""
    devices = sd.query_devices()
    inputs = []
    default_input = sd.default.device[0]
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            inputs.append({
                "index": idx,
                "name": d.get("name", f"Device #{idx}"),
                "channels": d.get("max_input_channels", 1),
                "default_samplerate": d.get("default_samplerate", 44100),
                "is_default": (idx == default_input),
            })
    return inputs


class CircularAudioBuffer:
    """Pre-allocated circular buffer for zero-allocation audio streaming.
    
    Prevents memory growth, buffer overruns, and garbage collection pauses.
    """

    def __init__(self, capacity_seconds: float = 15.0, sample_rate: int = SAMPLE_RATE) -> None:
        self.capacity = int(capacity_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_idx = 0
        self.total_written = 0
        self._lock = threading.Lock()

    def write(self, samples: np.ndarray) -> None:
        """Write new audio samples into circular buffer."""
        n = len(samples)
        if n == 0:
            return

        with self._lock:
            if n >= self.capacity:
                # Samples exceed capacity: keep only the newest window
                self.buffer[:] = samples[-self.capacity:]
                self.write_idx = 0
                self.total_written += n
                return

            end_idx = self.write_idx + n
            if end_idx <= self.capacity:
                self.buffer[self.write_idx:end_idx] = samples
            else:
                first_part = self.capacity - self.write_idx
                self.buffer[self.write_idx:] = samples[:first_part]
                self.buffer[: n - first_part] = samples[first_part:]

            self.write_idx = (self.write_idx + n) % self.capacity
            self.total_written += n

    def get_recent(self, duration_seconds: float) -> np.ndarray:
        """Extract the most recent N seconds of audio in linear order."""
        num_samples = min(int(duration_seconds * self.sample_rate), self.total_written, self.capacity)
        if num_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        with self._lock:
            start_idx = (self.write_idx - num_samples) % self.capacity
            if start_idx + num_samples <= self.capacity:
                return self.buffer[start_idx : start_idx + num_samples].copy()
            else:
                first_part = self.capacity - start_idx
                return np.concatenate((
                    self.buffer[start_idx:],
                    self.buffer[: num_samples - first_part],
                ))

    def clear(self) -> None:
        with self._lock:
            self.buffer.fill(0)
            self.write_idx = 0
            self.total_written = 0


class AudioStreamer:
    """Robust audio capture with Circular Buffer and Watchdog Auto-Recovery."""

    def __init__(
        self,
        device_index: Optional[int] = None,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = BLOCK_SIZE,
        buffer_seconds: float = 15.0,
    ) -> None:
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.buffer = CircularAudioBuffer(capacity_seconds=buffer_seconds, sample_rate=sample_rate)

        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()
        self._last_audio_timestamp = 0.0
        self._watchdog_thread: Optional[threading.Thread] = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        # Zero-allocation high-priority audio callback
        self._last_audio_timestamp = time.time()
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        self.buffer.write(mono)

    def _init_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device_index,
                channels=CHANNELS,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
            self._last_audio_timestamp = time.time()
        except Exception as e:
            raise MicrophoneAccessError(
                f"Cannot open microphone input: {e}\n"
                "  Check Windows Settings -> Privacy -> Microphone ('Let apps access your microphone').\n"
                "  Run 'autocut devices' to choose a valid input device."
            ) from e

    def _watchdog_loop(self) -> None:
        """Monitors audio stream health and re-establishes broken or stalled streams."""
        while self._running:
            time.sleep(0.5)
            now = time.time()
            # If running, but no audio arrived for > 2.5 seconds, auto-recover stream
            if self._running and (now - self._last_audio_timestamp) > 2.5:
                logger.warning("Audio input stalled or device disconnected. Attempting auto-recovery...")
                with self._lock:
                    if not self._running:
                        break
                    try:
                        self._init_stream()
                        logger.info("Audio stream successfully reconnected.")
                    except Exception as e:
                        logger.error("Audio reconnect failed: %s. Retrying in 1s...", e)
                        time.sleep(1.0)

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._init_stream()
            self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog_thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def get_recent_audio(self, duration_seconds: float) -> np.ndarray:
        return self.buffer.get_recent(duration_seconds)

    def is_active(self) -> bool:
        return self._running


class MicrophoneRecorder:
    """Records microphone audio into a contiguous 16kHz float32 numpy array."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        device_index: Optional[int] = None,
        on_chunk: Optional[Callable[[np.ndarray], None]] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.on_chunk = on_chunk
        self._chunks: list[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._recording = False
        self._lock = threading.Lock()

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if indata.ndim > 1 and indata.shape[1] > 1:
            mono = np.mean(indata, axis=1)
        else:
            mono = indata.flatten()
        copy_chunk = mono.copy()
        with self._lock:
            if self._recording:
                self._chunks.append(copy_chunk)
        if self.on_chunk and self._recording:
            try:
                self.on_chunk(copy_chunk)
            except Exception:
                pass

    def start(self) -> None:
        """Start capturing microphone audio."""
        with self._lock:
            self._chunks.clear()
            self._recording = True
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=CHANNELS,
                device=self.device_index,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as e:
            with self._lock:
                self._recording = False
            raise MicrophoneAccessError(
                f"Cannot access microphone device: {e}\n"
                "  Check Windows Settings -> Privacy -> Microphone ('Let desktop apps access your microphone').\n"
                "  Run 'autocut devices' to choose a valid input device."
            ) from e

    def stop(self) -> np.ndarray:
        """Stop capturing and return the recorded audio as float32 numpy array."""
        with self._lock:
            self._recording = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(self._chunks)

    @property
    def is_recording(self) -> bool:
        return self._recording
