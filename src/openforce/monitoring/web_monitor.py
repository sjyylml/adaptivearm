"""Web-based real-time monitoring for robot state and observer outputs.

Uses only the Python standard library (http.server, threading, json) to
provide a lightweight HTTP server with JSON API and an embedded HTML dashboard.
"""

from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import numpy as np
from numpy.typing import NDArray

from openforce.core.robot_state import RobotState
from openforce.core.types import ControlOutput, ObserverOutput


@dataclass
class WebMonitorConfig:
    """Configuration for the web monitor.

    Attributes:
        host: Server bind address.
        port: Server port.
        buffer_size: Maximum number of snapshots to store.
        page_title: Title for the HTML dashboard.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    buffer_size: int = 5000
    page_title: str = "OpenForce Monitor"


@dataclass
class MonitorSnapshot:
    """A single data snapshot for monitoring.

    Attributes:
        timestamp: Time in seconds.
        q: Joint positions.
        qd: Joint velocities.
        tau_motor: Motor torques.
        tau_ext: Estimated external torques (from observer).
        tau_cmd: Commanded torques (from controller).
        extra: Additional user-defined data.
    """

    timestamp: float
    q: NDArray[np.floating]
    qd: NDArray[np.floating]
    tau_motor: NDArray[np.floating]
    tau_ext: NDArray[np.floating] | None = None
    tau_cmd: NDArray[np.floating] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "q": self.q.tolist(),
            "qd": self.qd.tolist(),
            "tau_motor": self.tau_motor.tolist(),
        }
        if self.tau_ext is not None:
            d["tau_ext"] = self.tau_ext.tolist()
        if self.tau_cmd is not None:
            d["tau_cmd"] = self.tau_cmd.tolist()
        if self.extra:
            d["extra"] = self.extra
        return d


_DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #eee; margin: 20px; }}
h1 {{ color: #e94560; }}
.chart-container {{ width: 100%; max-width: 900px; margin: 20px auto; }}
canvas {{ background: #16213e; border-radius: 8px; }}
#status {{ color: #0f3460; font-size: 14px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p id="status">Fetching data...</p>
<div class="chart-container"><canvas id="torqueChart"></canvas></div>
<div class="chart-container"><canvas id="posChart"></canvas></div>
<script>
const torqueCtx = document.getElementById('torqueChart').getContext('2d');
const posCtx = document.getElementById('posChart').getContext('2d');
const colors = ['#e94560','#0f3460','#533483','#16c79a','#f5a623','#50c1e9'];

let torqueChart = new Chart(torqueCtx, {{
  type: 'line', data: {{ datasets: [] }},
  options: {{ responsive: true, animation: false,
    plugins: {{ title: {{ display: true, text: 'External Torques (Nm)', color: '#eee' }} }},
    scales: {{ x: {{ title: {{ display: true, text: 'Time (s)', color: '#eee' }} }} }}
  }}
}});
let posChart = new Chart(posCtx, {{
  type: 'line', data: {{ datasets: [] }},
  options: {{ responsive: true, animation: false,
    plugins: {{ title: {{ display: true, text: 'Joint Positions (rad)', color: '#eee' }} }},
    scales: {{ x: {{ title: {{ display: true, text: 'Time (s)', color: '#eee' }} }} }}
  }}
}});

async function update() {{
  try {{
    const resp = await fetch('/api/data');
    const data = await resp.json();
    if (!data.length) return;
    const times = data.map(d => d.timestamp.toFixed(3));
    const nj = data[0].q.length;

    torqueChart.data.labels = times;
    torqueChart.data.datasets = [];
    if (data[0].tau_ext) {{
      for (let j = 0; j < nj; j++) {{
        torqueChart.data.datasets.push({{
          label: 'J' + j, data: data.map(d => d.tau_ext[j]),
          borderColor: colors[j % colors.length], fill: false, pointRadius: 0
        }});
      }}
    }}
    torqueChart.update();

    posChart.data.labels = times;
    posChart.data.datasets = [];
    for (let j = 0; j < nj; j++) {{
      posChart.data.datasets.push({{
        label: 'J' + j, data: data.map(d => d.q[j]),
        borderColor: colors[j % colors.length], fill: false, pointRadius: 0
      }});
    }}
    posChart.update();
    document.getElementById('status').textContent =
      'Samples: ' + data.length + ' | Last t=' + data[data.length-1].timestamp.toFixed(3) + 's';
  }} catch(e) {{ document.getElementById('status').textContent = 'Error: ' + e; }}
}}
setInterval(update, 500);
update();
</script>
</body>
</html>"""


class _MonitorHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the web monitor."""

    monitor: WebMonitor  # Set via partial class creation

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_html()
        elif self.path == "/api/data":
            self._send_json(self.monitor.get_data())
        elif self.path == "/api/latest":
            latest = self.monitor.get_latest()
            if latest is not None:
                self._send_json([latest.to_dict()])
            else:
                self._send_json([])
        else:
            self.send_error(404)

    def _send_html(self) -> None:
        html = _DASHBOARD_HTML.format(title=self.monitor._config.page_title)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _send_json(self, data: list[dict[str, Any]] | list[Any]) -> None:
        body = json.dumps(data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""


class WebMonitor:
    """Web-based real-time monitoring server.

    Collects robot state snapshots and serves them via HTTP with a JSON API
    and an embedded Chart.js dashboard.

    Usage::

        with WebMonitor(WebMonitorConfig(port=8080)) as monitor:
            for _ in range(1000):
                state = adapter.get_state()
                output = observer.update(state)
                monitor.log(state, output)
            # Dashboard available at http://localhost:8080

    Args:
        config: Monitor configuration.
    """

    def __init__(self, config: WebMonitorConfig | None = None) -> None:
        self._config = config or WebMonitorConfig()
        self._buffer: deque[MonitorSnapshot] = deque(
            maxlen=self._config.buffer_size
        )
        self._lock = threading.Lock()
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def log(
        self,
        state: RobotState,
        observer_output: ObserverOutput | None = None,
        control_output: ControlOutput | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a data snapshot.

        Thread-safe. Can be called from the control loop.

        Args:
            state: Current robot state.
            observer_output: Optional observer output.
            control_output: Optional controller output.
            extra: Optional additional data.
        """
        snapshot = MonitorSnapshot(
            timestamp=state.timestamp,
            q=state.q.copy(),
            qd=state.qd.copy(),
            tau_motor=state.tau_motor.copy(),
            tau_ext=(
                observer_output.tau_ext.copy() if observer_output is not None else None
            ),
            tau_cmd=(
                control_output.tau_cmd.copy() if control_output is not None else None
            ),
            extra=extra or {},
        )
        with self._lock:
            self._buffer.append(snapshot)

    def get_data(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Get stored snapshots as JSON-serializable dicts.

        Args:
            last_n: If provided, return only the last N snapshots.

        Returns:
            List of snapshot dictionaries.
        """
        with self._lock:
            items = list(self._buffer)[-last_n:] if last_n is not None else list(self._buffer)
        return [s.to_dict() for s in items]

    def get_latest(self) -> MonitorSnapshot | None:
        """Get the most recent snapshot.

        Returns:
            Latest snapshot or None if buffer is empty.
        """
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def clear(self) -> None:
        """Clear the data buffer."""
        with self._lock:
            self._buffer.clear()

    def start(self) -> None:
        """Start the HTTP server in a background daemon thread."""
        if self._server is not None:
            return

        # Create handler class with reference to this monitor
        handler_class = type(
            "_BoundHandler",
            (_MonitorHandler,),
            {"monitor": self},
        )

        self._server = HTTPServer(
            (self._config.host, self._config.port),
            handler_class,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server is not None:
            self._server.shutdown()
            self._server = None
            self._thread = None

    def __enter__(self) -> WebMonitor:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
