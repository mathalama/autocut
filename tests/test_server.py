import asyncio
import time
import urllib.request
import websockets
from autocut.server import OverlayServer


def test_unified_overlay_server():
    server = OverlayServer(port=8925)
    server.start()
    try:
        time.sleep(0.3)
        # Test HTTP GET
        with urllib.request.urlopen("http://127.0.0.1:8925/index.html") as response:
            assert response.status == 200
            content = response.read().decode("utf-8")
            assert "AutoCut Live Captions" in content

        # Test WebSocket connection to same port
        async def test_ws():
            async with websockets.connect("ws://127.0.0.1:8925/ws") as ws:
                await asyncio.sleep(0.15)
                server.broadcast({"type": "caption", "text": "Testing", "is_final": False})
                msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                assert "Testing" in msg

        asyncio.run(test_ws())

        # Test abrupt TCP disconnection (browser preconnect / 0-byte drop)
        import socket
        s = socket.socket()
        s.connect(("127.0.0.1", 8925))
        s.close()
        time.sleep(0.1)
    finally:
        server.stop()

