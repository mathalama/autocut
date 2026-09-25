"""Unified HTTP and WebSocket server for OBS Browser Source overlay on a single port."""

import asyncio
import json
import logging
from mimetypes import guess_type
from pathlib import Path
import threading
from typing import Any, Optional, Set
import websockets
from websockets.datastructures import Headers
from websockets.http11 import Request, Response

from autocut.config import DEFAULT_HOST, DEFAULT_PORT, STATIC_DIR

logger = logging.getLogger(__name__)


class HandshakeNoiseFilter(logging.Filter):
    """Filter out noisy tracebacks from aborted TCP connections (browser pre-connects, zero-byte probes)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if "opening handshake failed" in record.getMessage() and record.exc_info:
            _, exc_val, _ = record.exc_info
            if isinstance(exc_val, (EOFError, ConnectionResetError)):
                return False
            if exc_val is not None:
                cause = getattr(exc_val, "__cause__", None)
                if isinstance(cause, (EOFError, ConnectionResetError)):
                    return False
                msg = str(exc_val).lower()
                cause_msg = str(cause).lower() if cause else ""
                if (
                    "did not receive a valid http request" in msg
                    or "stream ends after 0 bytes" in cause_msg
                    or "connection closed" in cause_msg
                ):
                    return False
        return True


# Suppress handshake tracebacks for empty TCP connections
logging.getLogger("websockets.server").addFilter(HandshakeNoiseFilter())


class OverlayServer:
    """Unified server that serves HTTP static files and WebSocket captions on the same port."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        static_dir: Path = STATIC_DIR,
        **kwargs: Any,
    ) -> None:
        self.host = host
        # Seamlessly support legacy http_port or ws_port arguments
        self.port = kwargs.get("http_port") or kwargs.get("port") or port
        self.static_dir = static_dir

        self._clients: Set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._running = False

    def _handle_http_request(self, connection: Any, request: Request) -> Optional[Response]:
        """Handle HTTP requests or return None to proceed with WebSocket upgrade."""
        # Strip query parameters from path (e.g. /?theme=glass -> /)
        path = request.path.split("?")[0].strip()

        # If requesting WebSocket endpoint, return None to proceed with WS handshake
        if path == "/ws":
            return None

        # Handle favicon without 404 noise
        if path == "/favicon.ico":
            return Response(204, "No Content", Headers([("Content-Length", "0")]), b"")

        # Map root to index.html
        relative_path = "index.html" if path in ("", "/") else path.lstrip("/")
        target_file = (self.static_dir / relative_path).resolve()

        # Security check: ensure target_file is inside static_dir
        if not str(target_file).startswith(str(self.static_dir.resolve())):
            return Response(403, "Forbidden", Headers([("Content-Type", "text/plain")]), b"Forbidden")

        if target_file.is_file():
            content_type, _ = guess_type(str(target_file))
            content_type = content_type or "application/octet-stream"
            body = target_file.read_bytes()
            headers = Headers([
                ("Content-Type", content_type),
                ("Content-Length", str(len(body))),
                ("Access-Control-Allow-Origin", "*"),
                ("Cache-Control", "no-cache"),
            ])
            return Response(200, "OK", headers, body)

        return Response(404, "Not Found", Headers([("Content-Type", "text/plain")]), b"Not Found")

    async def _ws_handler(self, websocket: Any) -> None:
        """Handle active WebSocket clients."""
        self._clients.add(websocket)
        try:
            async for _ in websocket:
                pass
        except Exception:
            pass
        finally:
            self._clients.discard(websocket)

    def _run_server(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def serve_all():
            try:
                async with websockets.serve(
                    self._ws_handler,
                    self.host,
                    self.port,
                    process_request=self._handle_http_request,
                ) as server:
                    self._server = server
                    while self._running:
                        await asyncio.sleep(0.15)
            except Exception as e:
                logger.debug("Server loop ended: %s", e)

        try:
            self._loop.run_until_complete(serve_all())
        except Exception as e:
            logger.debug("Event loop stopped: %s", e)
        finally:
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                if pending:
                    self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                self._loop.close()
            except Exception:
                pass

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

    def broadcast(self, payload: dict[str, Any]) -> None:
        """Thread-safe non-blocking broadcast with timeout protection."""
        if not self._running or not self._clients or not self._loop:
            return

        message = json.dumps(payload)

        async def _safe_send(client: Any) -> None:
            try:
                await asyncio.wait_for(client.send(message), timeout=0.15)
            except Exception:
                self._clients.discard(client)

        async def _send_all() -> None:
            if not self._clients:
                return
            await asyncio.gather(
                *[_safe_send(c) for c in list(self._clients)],
                return_exceptions=True,
            )

        asyncio.run_coroutine_threadsafe(_send_all(), self._loop)

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
