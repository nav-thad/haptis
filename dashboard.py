#!/usr/bin/env python3
"""
ESP32 Sound Localization Dashboard
Parses serial output and visualizes sound classification + haptic triggers on a compass.

Usage:
    python dashboard.py                    # opens browser at http://localhost:8050
    python dashboard.py --demo             # replay built-in demo data
    python dashboard.py --file output.log  # replay a saved serial log file
"""

import argparse
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime

import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_lock = threading.Lock()
_state = {
    "connected": False,
    "port": None,
    "azimuth": 0.0,
    "noise": 0.0,
    "voice": 0.0,
    "alarm": 0.0,
    "best_class": "—",
    "haptic_events": deque(maxlen=200),
    "serial_log": deque(maxlen=300),
    "inference_ms": None,
    "mfcc_ms": None,
    "last_haptic": None,
}

_serial_conn: serial.Serial | None = None
_reader_thread: threading.Thread | None = None

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
RE_INFERENCE = re.compile(
    r'\[Noise:\s*([\d.]+)\s*\|\s*Voice:\s*([\d.]+)\s*\|\s*Alarm:\s*([\d.]+)\]\s*→\s*(\w+)'
)
RE_HAPTIC    = re.compile(r'HAPTIC:\s*(\w+)\s+at\s+([\d.]+)°')
RE_AZIMUTH   = re.compile(r'Azimuth:\s*([\d.]+)°')
RE_MFCC_MS   = re.compile(r'MFCC:\s*([\d.]+)\s*ms')
RE_INF_MS    = re.compile(r'Inference:\s*([\d.]+)\s*ms')


def _parse_line(line: str) -> None:
    with _lock:
        _state["serial_log"].appendleft(line)

    m = RE_INFERENCE.search(line)
    if m:
        with _lock:
            _state["noise"]      = float(m.group(1))
            _state["voice"]      = float(m.group(2))
            _state["alarm"]      = float(m.group(3))
            _state["best_class"] = m.group(4)
        return

    m = RE_HAPTIC.search(line)
    if m:
        ev = {
            "time":  datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "type":  m.group(1),
            "angle": float(m.group(2)),
        }
        with _lock:
            _state["haptic_events"].appendleft(ev)
            _state["azimuth"]    = ev["angle"]
            _state["last_haptic"] = ev
        return

    m = RE_AZIMUTH.search(line)
    if m:
        with _lock:
            _state["azimuth"] = float(m.group(1))
        return

    m = RE_MFCC_MS.search(line)
    if m:
        with _lock:
            _state["mfcc_ms"] = float(m.group(1))
        return

    m = RE_INF_MS.search(line)
    if m:
        with _lock:
            _state["inference_ms"] = float(m.group(1))
        return


# ---------------------------------------------------------------------------
# Serial reader thread
# ---------------------------------------------------------------------------
def _serial_reader(port: str, baud: int = 921600) -> None:
    global _serial_conn
    try:
        _serial_conn = serial.Serial(port, baud, timeout=1)
        with _lock:
            _state["connected"] = True
            _state["port"]      = port
        while True:
            with _lock:
                if not _state["connected"]:
                    break
            try:
                raw = _serial_conn.readline()
                line = raw.decode("utf-8", errors="replace").strip()
                if line:
                    _parse_line(line)
            except Exception:
                break
    except Exception as exc:
        with _lock:
            _state["serial_log"].appendleft(f"[ERROR] {exc}")
    finally:
        if _serial_conn and _serial_conn.is_open:
            _serial_conn.close()
        with _lock:
            _state["connected"] = False


# ---------------------------------------------------------------------------
# Demo / file replay thread
# ---------------------------------------------------------------------------
DEMO_LINES = [
    "ESP32-S3 | Sound Localization + Classification + Haptics (Dual-Core)",
    "I2S0 OK  (Mics 0 & 1 — stereo)",
    "I2S1 OK  (Mic 2 — mono)",
    "Model loaded | Arena used: 145200 / 200000 bytes",
    "Ready.",
    "MFCC: 42 ms | MFCC range: -3.21, 4.87 mean: 0.02",
    "Inference: 18 ms",
    "Azimuth: 45°",
    "[Noise: 0.03 | Voice: 0.92 | Alarm: 0.05] → Voice",
    "  ► HAPTIC: Voice at 45°",
    "MFCC: 39 ms | MFCC range: -2.98, 5.10 mean: 0.01",
    "Inference: 17 ms",
    "Azimuth: 90°",
    "[Noise: 0.88 | Voice: 0.08 | Alarm: 0.04] → Noise",
    "MFCC: 41 ms | MFCC range: -3.00, 4.60 mean: -0.01",
    "Inference: 19 ms",
    "Azimuth: 135°",
    "[Noise: 0.04 | Voice: 0.05 | Alarm: 0.91] → Alarm",
    "  ► HAPTIC: Alarm at 135°",
    "MFCC: 40 ms | MFCC range: -3.10, 4.75 mean: 0.00",
    "Inference: 16 ms",
    "Azimuth: 200°",
    "[Noise: 0.71 | Voice: 0.24 | Alarm: 0.05] → Noise",
    "MFCC: 38 ms | MFCC range: -2.80, 4.90 mean: 0.02",
    "Inference: 18 ms",
    "Azimuth: 310°",
    "[Noise: 0.02 | Voice: 0.96 | Alarm: 0.02] → Voice",
    "  ► HAPTIC: Voice at 310°",
]


def _demo_reader() -> None:
    with _lock:
        _state["connected"] = True
        _state["port"]      = "DEMO"
    idx = 0
    while True:
        with _lock:
            if not _state["connected"]:
                break
        line = DEMO_LINES[idx % len(DEMO_LINES)]
        _parse_line(line)
        idx += 1
        time.sleep(0.8)


def _file_reader(path: str) -> None:
    with _lock:
        _state["connected"] = True
        _state["port"]      = path
    try:
        with open(path, "r", errors="replace") as f:
            lines = f.readlines()
        for line in lines:
            with _lock:
                if not _state["connected"]:
                    break
            _parse_line(line.strip())
            time.sleep(0.05)
    except Exception as exc:
        with _lock:
            _state["serial_log"].appendleft(f"[ERROR] {exc}")
    finally:
        with _lock:
            _state["connected"] = False


# ---------------------------------------------------------------------------
# Dash layout
# ---------------------------------------------------------------------------
CARD = {
    "background": "#ffffff",
    "borderRadius": "8px",
    "boxShadow": "0 1px 4px rgba(0,0,0,0.10)",
    "padding": "12px",
}

app = dash.Dash(__name__, title="ESP32 Sound Dashboard")
app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "background": "#f0f2f5", "minHeight": "100vh"},
    children=[
        # ── Header ──────────────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex", "alignItems": "center",
                "padding": "12px 24px", "background": "#1a1a2e", "color": "white",
                "gap": "12px",
            },
            children=[
                html.Div("🔊", style={"fontSize": "22px"}),
                html.H2("ESP32 Sound Localization Dashboard",
                        style={"margin": "0", "fontSize": "18px", "fontWeight": "600"}),
                html.Div(id="status-badge", style={"marginLeft": "auto"}),
            ],
        ),

        # ── Controls bar ────────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex", "alignItems": "center", "flexWrap": "wrap",
                "padding": "10px 24px", "background": "#e8eaf0",
                "gap": "8px", "borderBottom": "1px solid #d0d4de",
            },
            children=[
                dcc.Dropdown(
                    id="port-dropdown",
                    placeholder="Select Serial Port…",
                    style={"width": "280px", "color": "#333"},
                ),
                html.Button("↺ Refresh",    id="refresh-btn",    n_clicks=0,
                            style={"padding": "6px 14px", "borderRadius": "5px",
                                   "border": "1px solid #ccc", "cursor": "pointer",
                                   "fontSize": "13px", "background": "#fff"}),
                html.Button("⚡ Connect",    id="connect-btn",    n_clicks=0,
                            style={"padding": "6px 14px", "borderRadius": "5px",
                                   "border": "1px solid #219a52", "cursor": "pointer",
                                   "fontSize": "13px", "background": "#27ae60",
                                   "color": "white"}),
                html.Button("✖ Disconnect", id="disconnect-btn", n_clicks=0,
                            style={"padding": "6px 14px", "borderRadius": "5px",
                                   "border": "1px solid #c0392b", "cursor": "pointer",
                                   "fontSize": "13px", "background": "#e74c3c",
                                   "color": "white"}),
                html.Span("Baud: 921600", style={"color": "#666", "fontSize": "12px",
                                                  "marginLeft": "8px"}),
            ],
        ),

        # ── Main grid ───────────────────────────────────────────────────────
        html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "1fr 1fr",
                   "gridTemplateRows": "auto auto",
                   "gap": "16px", "padding": "16px 24px"},
            children=[
                # Compass
                html.Div(style={**CARD, "gridColumn": "1", "gridRow": "1"},
                         children=[dcc.Graph(id="compass-plot",
                                             style={"height": "400px"},
                                             config={"displayModeBar": False})]),

                # Probability bars
                html.Div(style={**CARD, "gridColumn": "2", "gridRow": "1"},
                         children=[dcc.Graph(id="prob-bars",
                                             style={"height": "400px"},
                                             config={"displayModeBar": False})]),

                # Haptic event log
                html.Div(
                    style={**CARD, "gridColumn": "1", "gridRow": "2"},
                    children=[
                        html.H4("Haptic Events",
                                style={"margin": "0 0 10px", "fontSize": "14px",
                                       "fontWeight": "600", "color": "#333"}),
                        html.Div(id="haptic-table",
                                 style={"height": "220px", "overflowY": "auto",
                                        "fontFamily": "monospace", "fontSize": "12px"}),
                    ],
                ),

                # Serial log
                html.Div(
                    style={**CARD, "gridColumn": "2", "gridRow": "2"},
                    children=[
                        html.H4("Serial Log",
                                style={"margin": "0 0 10px", "fontSize": "14px",
                                       "fontWeight": "600", "color": "#333"}),
                        html.Div(id="serial-log",
                                 style={"height": "220px", "overflowY": "auto",
                                        "background": "#0d1117", "color": "#58d96a",
                                        "borderRadius": "6px", "padding": "8px",
                                        "fontFamily": "monospace", "fontSize": "11px",
                                        "lineHeight": "1.5"}),
                    ],
                ),
            ],
        ),

        # Polling interval
        dcc.Interval(id="interval", interval=250, n_intervals=0),
    ],
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output("port-dropdown", "options"),
    Input("refresh-btn", "n_clicks"),
)
def refresh_ports(_):
    ports = serial.tools.list_ports.comports()
    opts = [{"label": f"{p.device}  —  {p.description}", "value": p.device}
            for p in ports]
    if not opts:
        opts = [{"label": "(no serial ports found)", "value": "", "disabled": True}]
    return opts


@app.callback(
    Output("status-badge", "children"),
    Output("status-badge", "style"),
    Input("interval", "n_intervals"),
)
def update_status(_):
    with _lock:
        connected = _state["connected"]
        port      = _state["port"]
    text  = f"● {port}" if connected else "● Disconnected"
    color = "#27ae60" if connected else "#e74c3c"
    style = {
        "padding": "4px 14px", "borderRadius": "20px",
        "background": color, "color": "white",
        "fontSize": "12px", "fontWeight": "600",
    }
    return text, style


@app.callback(
    Output("port-dropdown", "value"),
    Input("connect-btn",    "n_clicks"),
    Input("disconnect-btn", "n_clicks"),
    State("port-dropdown",  "value"),
    prevent_initial_call=True,
)
def handle_connection(_conn, _disc, port):
    global _reader_thread, _serial_conn
    ctx     = callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0]

    if trigger == "connect-btn" and port:
        with _lock:
            if _state["connected"]:
                return port
        _reader_thread = threading.Thread(
            target=_serial_reader, args=(port,), daemon=True
        )
        _reader_thread.start()

    elif trigger == "disconnect-btn":
        with _lock:
            _state["connected"] = False
        if _serial_conn and _serial_conn.is_open:
            _serial_conn.close()

    return port


@app.callback(
    Output("compass-plot", "figure"),
    Input("interval", "n_intervals"),
)
def update_compass(_):
    with _lock:
        azimuth = _state["azimuth"]
        best    = _state["best_class"]
        events  = list(_state["haptic_events"])

    fig = go.Figure()

    # ── Recent haptic dots (fading) ──────────────────────────────────────
    n = min(len(events), 15)
    for i, ev in enumerate(reversed(events[:n])):
        alpha = round((i + 1) / n, 2)
        if ev["type"] == "Alarm":
            rgba = f"rgba(231,76,60,{alpha})"
        else:
            rgba = f"rgba(39,174,96,{alpha})"

        fig.add_trace(go.Scatterpolar(
            r=[0.80], theta=[ev["angle"]],
            mode="markers",
            marker=dict(size=10, color=rgba, symbol="circle"),
            name=f'{ev["type"]} {ev["angle"]:.0f}° {ev["time"]}',
            showlegend=(i == n - 1),
            hovertemplate=f'{ev["type"]} · {ev["angle"]:.1f}° · {ev["time"]}<extra></extra>',
        ))

    # ── Direction needle ────────────────────────────────────────────────
    needle_color = (
        "#e74c3c" if best == "Alarm" else
        "#27ae60" if best == "Voice" else
        "#3498db"
    )
    fig.add_trace(go.Scatterpolar(
        r=[0, 0.95], theta=[azimuth, azimuth],
        mode="lines+markers",
        line=dict(color=needle_color, width=5),
        marker=dict(size=[0, 10], color=needle_color, symbol=["circle", "arrow"],
                    angleref="previous"),
        name=f"{best} @ {azimuth:.1f}°",
        showlegend=True,
        hovertemplate=f"Azimuth: {azimuth:.1f}°<extra></extra>",
    ))

    # ── Origin dot ──────────────────────────────────────────────────────
    fig.add_trace(go.Scatterpolar(
        r=[0], theta=[0], mode="markers",
        marker=dict(size=8, color="#444"), showlegend=False,
    ))

    fig.update_layout(
        polar=dict(
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=["N  0°", "NE  45°", "E  90°", "SE  135°",
                          "S  180°", "SW  225°", "W  270°", "NW  315°"],
                direction="clockwise",
                rotation=90,
                gridcolor="#d5dae3",
                linecolor="#c0c8d8",
                tickfont=dict(size=11),
            ),
            radialaxis=dict(visible=False, range=[0, 1.1]),
            bgcolor="#f7f9fc",
        ),
        title=dict(
            text=f"<b>Sound Direction</b>   {azimuth:.1f}°  ·  {best}",
            font=dict(size=14, color="#222"),
            x=0.5, xanchor="center",
        ),
        showlegend=True,
        legend=dict(
            x=1.08, y=0.95, bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#ddd", borderwidth=1, font=dict(size=11),
        ),
        margin=dict(l=40, r=150, t=55, b=40),
        paper_bgcolor="#ffffff",
    )
    return fig


@app.callback(
    Output("prob-bars", "figure"),
    Input("interval", "n_intervals"),
)
def update_probs(_):
    with _lock:
        noise    = _state["noise"]
        voice    = _state["voice"]
        alarm    = _state["alarm"]
        best     = _state["best_class"]
        inf_ms   = _state["inference_ms"]
        mfcc_ms  = _state["mfcc_ms"]
        last_ev  = _state["last_haptic"]

    classes = ["Noise", "Voice", "Alarm"]
    vals    = [noise, voice, alarm]
    colors  = ["#5b8dee", "#27ae60", "#e74c3c"]
    borders = [4 if c == best else 0 for c in classes]

    # Threshold lines
    THRESH = 0.85

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=classes, y=vals,
        marker_color=colors,
        marker_line_width=borders,
        marker_line_color="#111",
        text=[f"{v:.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=13, color="#333"),
        width=0.45,
    ))

    # Threshold indicator line
    fig.add_shape(
        type="line", x0=-0.5, x1=2.5, y0=THRESH, y1=THRESH,
        line=dict(color="#e67e22", width=2, dash="dot"),
    )
    fig.add_annotation(
        x=2.55, y=THRESH, text=f"threshold {THRESH}",
        showarrow=False, font=dict(size=10, color="#e67e22"),
        xanchor="left",
    )

    # Last haptic annotation
    if last_ev:
        fig.add_annotation(
            x=0.5, y=1.08,
            text=f"Last haptic: <b>{last_ev['type']}</b> @ {last_ev['angle']:.0f}°  [{last_ev['time']}]",
            showarrow=False,
            font=dict(size=11, color="#c0392b" if last_ev["type"] == "Alarm" else "#27ae60"),
            xref="paper", yref="paper",
        )

    timing = []
    if mfcc_ms is not None:
        timing.append(f"MFCC {mfcc_ms:.0f} ms")
    if inf_ms is not None:
        timing.append(f"Inf {inf_ms:.0f} ms")
    timing_str = "   ·   " + "   ·   ".join(timing) if timing else ""

    fig.update_layout(
        title=dict(
            text=f"<b>Class Probabilities</b>{timing_str}",
            font=dict(size=14, color="#222"),
            x=0.5, xanchor="center",
        ),
        yaxis=dict(range=[0, 1.25], gridcolor="#e8ecf2",
                   tickformat=".2f", title="Confidence"),
        xaxis=dict(gridcolor="#e8ecf2"),
        plot_bgcolor="#f7f9fc",
        paper_bgcolor="#ffffff",
        margin=dict(l=55, r=40, t=55, b=30),
        showlegend=False,
    )
    return fig


@app.callback(
    Output("haptic-table", "children"),
    Input("interval", "n_intervals"),
)
def update_haptic_table(_):
    with _lock:
        events = list(_state["haptic_events"])

    if not events:
        return html.Div("No haptic events yet.",
                        style={"color": "#aaa", "padding": "8px"})

    header = html.Div(
        style={"display": "grid", "gridTemplateColumns": "90px 60px 1fr",
               "padding": "4px 6px", "borderBottom": "2px solid #e0e4ed",
               "fontWeight": "600", "color": "#555", "fontSize": "11px"},
        children=[html.Span("Time"), html.Span("Type"), html.Span("Angle")],
    )

    rows = [header]
    for i, ev in enumerate(events):
        is_alarm = ev["type"] == "Alarm"
        color    = "#e74c3c" if is_alarm else "#27ae60"
        bg       = "#fff5f5" if (is_alarm and i == 0) else ("#f5fff8" if i == 0 else "transparent")
        rows.append(html.Div(
            style={"display": "grid", "gridTemplateColumns": "90px 60px 1fr",
                   "padding": "3px 6px", "borderBottom": "1px solid #f0f2f8",
                   "background": bg, "alignItems": "center"},
            children=[
                html.Span(ev["time"], style={"color": "#888"}),
                html.Span(ev["type"], style={"color": color, "fontWeight": "600"}),
                html.Span(f"{ev['angle']:.1f}°",
                          style={"color": "#333", "fontVariantNumeric": "tabular-nums"}),
            ],
        ))
    return rows


@app.callback(
    Output("serial-log", "children"),
    Input("interval", "n_intervals"),
)
def update_serial_log(_):
    with _lock:
        lines = list(_state["serial_log"])

    def line_color(l):
        if "HAPTIC" in l:    return "#f1c40f"
        if "ERROR" in l:     return "#e74c3c"
        if "Alarm" in l:     return "#e67e22"
        if "Voice" in l:     return "#2ecc71"
        if "Ready" in l:     return "#3498db"
        return "#58d96a"

    return [
        html.Div(line, style={"color": line_color(line), "margin": "0"})
        for line in lines[:80]
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    global _reader_thread

    parser = argparse.ArgumentParser(description="ESP32 Sound Dashboard")
    parser.add_argument("--demo", action="store_true",
                        help="Run with simulated demo data")
    parser.add_argument("--file", metavar="LOG",
                        help="Replay a saved serial log file")
    parser.add_argument("--port", metavar="PORT",
                        help="Auto-connect to this serial port on startup")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Dashboard host (default 127.0.0.1)")
    parser.add_argument("--dash-port", default=8050, type=int,
                        help="Dashboard port (default 8050)")
    args = parser.parse_args()

    if args.demo:
        print("Running in DEMO mode.")
        _reader_thread = threading.Thread(target=_demo_reader, daemon=True)
        _reader_thread.start()
    elif args.file:
        print(f"Replaying log file: {args.file}")
        _reader_thread = threading.Thread(
            target=_file_reader, args=(args.file,), daemon=True
        )
        _reader_thread.start()
    elif args.port:
        print(f"Auto-connecting to {args.port} @ 921600 baud…")
        _reader_thread = threading.Thread(
            target=_serial_reader, args=(args.port,), daemon=True
        )
        _reader_thread.start()

    print(f"Dashboard → http://{args.host}:{args.dash_port}/")
    app.run(debug=False, host=args.host, port=args.dash_port)


if __name__ == "__main__":
    main()
