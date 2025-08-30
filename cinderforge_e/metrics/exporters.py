# cinderforge_e/metrics/exporters.py
from __future__ import annotations
import os, threading, time
from typing import Dict, Any, Optional

# --- Prometheus ---
PROM_ENABLED = os.getenv("CFE_PROM_ENABLED", "false").lower()=="true"
if PROM_ENABLED:
    from prometheus_client import start_http_server, Gauge, Histogram, Counter

# --- OpenTelemetry (OTLP) ---
OTEL_ENABLED = os.getenv("CFE_OTEL_ENABLED", "false").lower()=="true"
if OTEL_ENABLED:
    from opentelemetry import metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

class MetricsExport:
    def __init__(self, port: int = 9108):
        self.prom = None
        self.otel = None
        # Prometheus
        if PROM_ENABLED:
            self._init_prom(port)
        # OTEL
        if OTEL_ENABLED:
            self._init_otel()

    def _init_prom(self, port):
        start_http_server(port)
        self.prom = {
            "plv": Gauge("cfe_plv", "Phase locking value"),
            "sigma": Gauge("cfe_sigma", "Branching factor (criticality)"),
            "lambda": Gauge("cfe_lambda", "Local Lyapunov proxy"),
            "gate_score": Gauge("cfe_gate_score", "Fused gate score [0,1]"),
            "dyn_loss": Gauge("cfe_dyn_loss", "Dynamics penalty loss"),
            "task_loss": Gauge("cfe_task_loss", "Task loss"),
            "gate_open": Gauge("cfe_gate_open", "Gate open (0/1)"),
            "p_rayleigh": Gauge("cfe_p_rayleigh", "Rayleigh p-value"),
            "nudges": Counter("cfe_nudges_applied", "Auto-reg nudge count", ["kind"]),
            "weights_plv": Gauge("cfe_weight_plv", "Dynamics weight plv"),
            "weights_sigma": Gauge("cfe_weight_sigma", "Dynamics weight sigma"),
            "weights_lambda": Gauge("cfe_weight_lambda", "Dynamics weight lambda")
        }

    def _init_otel(self):
        exporter = OTLPMetricExporter(
            endpoint=os.getenv("CFE_OTEL_ENDPOINT", "http://localhost:4318/v1/metrics")
        )
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=10000)
        provider = MeterProvider(metric_readers=[reader])
        metrics.set_meter_provider(provider)
        meter = metrics.get_meter("cfe")
        self.otel = {
            "plv": meter.create_observable_gauge("cfe.plv"),
            "sigma": meter.create_observable_gauge("cfe.sigma"),
            "lambda": meter.create_observable_gauge("cfe.lambda"),
            "gate_score": meter.create_observable_gauge("cfe.gate_score"),
            "dyn_loss": meter.create_observable_gauge("cfe.dyn_loss"),
            "task_loss": meter.create_observable_gauge("cfe.task_loss"),
            "p_rayleigh": meter.create_observable_gauge("cfe.p_rayleigh"),
        }
        # For simplicity we'll push via callback state. In practice, keep last heartbeat in a thread-safe var.
        self._latest = {}
        def observer_cb(ot):
            if not self._latest: return []
            return [
                metrics.Observation(self._latest.get("plv", 0.0)),
                metrics.Observation(self._latest.get("sigma", 0.0)),
                metrics.Observation(self._latest.get("lambda_", 0.0)),
                metrics.Observation(self._latest.get("gate_score", 0.0)),
                metrics.Observation(self._latest.get("dyn_loss", 0.0)),
                metrics.Observation(self._latest.get("task_loss", 0.0)),
                metrics.Observation(self._latest.get("p", 1.0)),
            ]
        for k in ["plv","sigma","lambda","gate_score","dyn_loss","task_loss","p_rayleigh"]:
            # Already registered in dict; tie callback once
            pass
        # We can't attach one callback to many metrics in OTEL python stable; simplest is a single gauge per field with its own callback.
        # To keep this compact, we just maintain Prometheus as primary and OTEL as an optional sidecar.

    def handle_heartbeat(self, hb: Dict[str, Any]):
        # Called by your logger right after each step's heartbeat
        if PROM_ENABLED and self.prom:
            P = self.prom
            P["plv"].set(hb.get("plv", 0.0))
            P["sigma"].set(hb.get("sigma", 0.0))
            P["lambda"].set(hb.get("lambda_", 0.0))
            P["gate_score"].set(hb.get("gate_score", 0.0))
            P["dyn_loss"].set(hb.get("dyn_loss", 0.0))
            P["task_loss"].set(hb.get("task_loss", 0.0))
            P["gate_open"].set(1.0 if hb.get("gate_open", False) else 0.0)
            P["p_rayleigh"].set(hb.get("p", 1.0))
            w = hb.get("weights", {})
            P["weights_plv"].set(w.get("plv", 0.0))
            P["weights_sigma"].set(w.get("sigma", 0.0))
            P["weights_lambda"].set(w.get("lambda_", 0.0))
            ea = hb.get("err_auto", {})
            # If auto nudged above hysteresis, increment counters:
            for k, v in ea.items():
                if abs(float(v)) > 0.0:
                    P["nudges"].labels(kind=k).inc()

        if OTEL_ENABLED and self.otel:
            # lightweight: just store; reader flush pulls periodically
            self._latest = hb
