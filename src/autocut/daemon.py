from __future__ import annotations

import io
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Optional
import numpy as np

from autocut.stt.pool import ModelPool
from autocut.stt.base import STTSegment

logger = logging.getLogger("autocut.daemon")
DEFAULT_DAEMON_PORT = 8766


class DaemonHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler for AutoCut background STT daemon."""

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress noisy HTTP access logs
        pass

    def _send_json(self, status_code: int, data: dict[str, Any]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/health":
            pool = ModelPool.instance()
            curr = pool._current_engine
            info = {
                "status": "ok",
                "loaded": curr.is_loaded if curr else False,
                "engine": curr.__class__.__name__ if curr else None,
                "model": curr.model_name if curr else None,
                "device": curr.device if curr else None,
            }
            self._send_json(200, info)
        else:
            self._send_json(404, {"error": "Not Found"})

    def do_POST(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        if self.path == "/transcribe":
            try:
                # Accept raw 16kHz float32 bytes directly
                audio = np.frombuffer(body, dtype=np.float32)
                pool = ModelPool.instance()
                engine = pool._current_engine
                if engine is None:
                    engine = pool.get_engine("whisper", "small.en")

                segments = engine.transcribe(audio, sample_rate=16000)
                data = {"segments": [s.to_dict() for s in segments]}
                self._send_json(200, data)
            except Exception as e:
                logger.error("Daemon transcribe error: %s", e)
                self._send_json(500, {"error": str(e)})

        elif self.path == "/transcribe_chunk":
            try:
                audio = np.frombuffer(body, dtype=np.float32)
                pool = ModelPool.instance()
                engine = pool._current_engine
                if engine is None:
                    engine = pool.get_engine("whisper", "small.en")

                text = engine.transcribe_chunk(audio, sample_rate=16000)
                self._send_json(200, {"text": text})
            except Exception as e:
                logger.error("Daemon transcribe_chunk error: %s", e)
                self._send_json(500, {"error": str(e)})

        elif self.path == "/switch_model":
            try:
                params = json.loads(body.decode("utf-8"))
                engine_type = params.get("engine", "whisper")
                model_name = params.get("model")
                device = params.get("device", "cuda")
                compute_type = params.get("compute_type", "float16")
                language = params.get("language")

                pool = ModelPool.instance()
                engine = pool.get_engine(
                    engine_type=engine_type,
                    model_name=model_name,
                    device=device,
                    compute_type=compute_type,
                    language=language,
                )
                self._send_json(200, {
                    "status": "ok",
                    "engine": engine.__class__.__name__,
                    "model": engine.model_name,
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Not Found"})


class STTDaemon:
    """Headless STT Daemon that keeps neural models pre-warmed in VRAM."""

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_DAEMON_PORT) -> None:
        self.host = host
        self.port = port
        self.server: Optional[HTTPServer] = None

    def start(self, engine_type: str = "whisper", model_name: Optional[str] = None, **kwargs: Any) -> None:
        # Pre-warm model in memory
        pool = ModelPool.instance()
        pool.get_engine(engine_type=engine_type, model_name=model_name, **kwargs)

        self.server = HTTPServer((self.host, self.port), DaemonHTTPHandler)
        logger.info("AutoCut STT Daemon running at http://%s:%d", self.host, self.port)
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        ModelPool.instance().unload_all()


class DaemonClient:
    """Client for communicating with local AutoCut STT Daemon."""

    def __init__(self, base_url: str = f"http://127.0.0.1:{DEFAULT_DAEMON_PORT}") -> None:
        self.base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        import urllib.request
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=0.6) as resp:
                return resp.status == 200
        except Exception:
            return False

    def transcribe(self, audio: np.ndarray) -> list[STTSegment]:
        import urllib.request
        audio_bytes = audio.astype(np.float32).tobytes()
        req = urllib.request.Request(
            f"{self.base_url}/transcribe",
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [
                STTSegment(
                    index=s["index"],
                    start=float(s["start"]),
                    end=float(s["end"]),
                    text=s["text"],
                )
                for s in data.get("segments", [])
            ]

    def transcribe_chunk(self, audio_chunk: np.ndarray) -> str:
        import urllib.request
        audio_bytes = audio_chunk.astype(np.float32).tobytes()
        req = urllib.request.Request(
            f"{self.base_url}/transcribe_chunk",
            data=audio_bytes,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("text", "")
