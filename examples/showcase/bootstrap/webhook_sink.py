"""Webhook sink: the convergence point for CI alert curls (runs in the
runner image; stdlib only, DESIGN.md section 3).

Records every POST body it receives; GET /requests returns them all as a
JSON list (newest last), each wrapped as {"path", "received_at", "body"}
with the body parsed as JSON when possible. Human-readable in demos,
assertable in tests. DELETE /requests resets between test phases.
"""

import json
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

RECEIVED: list = []


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status: int, payload: object) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/requests":
            self._reply(200, RECEIVED)
        else:
            self._reply(200, {"ok": True, "recorded": len(RECEIVED)})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode(errors="replace")
        try:
            body: object = json.loads(raw)
        except ValueError:
            body = raw
        entry = {
            "path": self.path,
            "received_at": datetime.now(UTC).isoformat(),
            "body": body,
        }
        RECEIVED.append(entry)
        print(f"webhook-sink: {self.path} <- {json.dumps(body)[:500]}", flush=True)
        self._reply(200, {"ok": True})

    def do_DELETE(self) -> None:
        if self.path.rstrip("/") == "/requests":
            RECEIVED.clear()
            self._reply(200, {"ok": True})
        else:
            self._reply(404, {"ok": False})

    def log_message(self, *_args: object) -> None:  # quiet: stdout is the record
        pass


if __name__ == "__main__":
    HTTPServer(("0.0.0.0", 9099), Handler).serve_forever()
