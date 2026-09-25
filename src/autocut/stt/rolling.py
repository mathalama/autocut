from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional
import numpy as np

from autocut.stt.base import BaseSTTEngine, STTSegment
from autocut.vad import DynamicNoiseFloorVAD

logger = logging.getLogger("autocut.stt.rolling")


class RollingTranscriber:
    """Pre-emptive rolling background transcription pipeline.
    
    Segments spoken audio on-the-fly and transcribes completed utterances
    asynchronously while the user continues speaking.
    """

    def __init__(
        self,
        engine: BaseSTTEngine,
        sample_rate: int = 16000,
        pause_threshold: float = 0.45,
        max_chunk_sec: float = 4.0,
        min_speech_sec: float = 0.35,
        energy_threshold: float = 0.015,
        on_segment: Optional[Callable[[STTSegment], None]] = None,
    ) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.pause_threshold = pause_threshold
        self.max_chunk_sec = max_chunk_sec
        self.min_speech_sec = min_speech_sec
        self.on_segment = on_segment

        self._vad = DynamicNoiseFloorVAD(energy_threshold=energy_threshold)
        self._work_queue: queue.Queue[tuple[np.ndarray, float, float, int]] = queue.Queue()
        self._results: list[STTSegment] = []
        self._results_lock = threading.Lock()

        self._current_speech_chunks: list[np.ndarray] = []
        self._speech_start_sec: float = 0.0
        self._total_processed_samples: int = 0
        self._silence_start_time: Optional[float] = None
        self._is_speaking = False
        self._segment_index = 1

        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background worker thread."""
        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="RollingSTTWorker")
        self._worker_thread.start()

    def process_audio_chunk(self, chunk: np.ndarray) -> None:
        """Feed a live audio frame (typically 50-100ms) from microphone."""
        chunk_len = len(chunk)
        chunk_sec = chunk_len / float(self.sample_rate)
        now_sample_offset = self._total_processed_samples
        self._total_processed_samples += chunk_len

        is_voice = self._vad.is_speech(chunk, sample_rate=self.sample_rate)

        if is_voice:
            self._silence_start_time = None
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start_sec = now_sample_offset / float(self.sample_rate)
                self._current_speech_chunks = []

            self._current_speech_chunks.append(chunk)

            # Enforce max duration per chunk to prevent delayed transcription on continuous speech
            accum_sec = sum(len(c) for c in self._current_speech_chunks) / float(self.sample_rate)
            if accum_sec >= self.max_chunk_sec:
                self._flush_current_speech(now_sample_offset / float(self.sample_rate))
                self._is_speaking = False
        else:
            if self._is_speaking:
                self._current_speech_chunks.append(chunk)
                now_sec = time.time()
                if self._silence_start_time is None:
                    self._silence_start_time = now_sec
                elif (now_sec - self._silence_start_time) >= self.pause_threshold:
                    # Speech boundary detected: emit utterance to queue
                    end_sec = (now_sample_offset + chunk_len) / float(self.sample_rate)
                    self._flush_current_speech(end_sec)
                    self._is_speaking = False
                    self._silence_start_time = None

    def _flush_current_speech(self, end_sec: float) -> None:
        if not self._current_speech_chunks:
            return

        audio_segment = np.concatenate(self._current_speech_chunks)
        self._current_speech_chunks = []
        duration = len(audio_segment) / float(self.sample_rate)

        if duration >= self.min_speech_sec:
            idx = self._segment_index
            self._segment_index += 1
            self._work_queue.put((audio_segment, self._speech_start_sec, end_sec, idx))

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set() or not self._work_queue.empty():
            try:
                item = self._work_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            audio_chunk, start_sec, end_sec, idx = item
            try:
                text = self.engine.transcribe_chunk(audio_chunk, sample_rate=self.sample_rate)
                if text:
                    seg = STTSegment(
                        index=idx,
                        start=round(start_sec, 2),
                        end=round(end_sec, 2),
                        text=text,
                    )
                    with self._results_lock:
                        self._results.append(seg)
                    if self.on_segment:
                        self.on_segment(seg)
            except Exception as e:
                logger.error("Rolling transcription error on segment %d: %s", idx, e)
            finally:
                self._work_queue.task_done()

    def finish(self) -> list[STTSegment]:
        """Stop intake, flush any pending speech, and wait for queue to drain."""
        if self._is_speaking and self._current_speech_chunks:
            end_sec = self._total_processed_samples / float(self.sample_rate)
            self._flush_current_speech(end_sec)
            self._is_speaking = False

        self._stop_event.set()
        # Wait for worker thread to process any queued segments
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        with self._results_lock:
            # Sort segments by start timestamp to ensure monotonic timeline
            sorted_results = sorted(self._results, key=lambda s: s.start)
            for i, s in enumerate(sorted_results, 1):
                s.index = i
            return sorted_results
