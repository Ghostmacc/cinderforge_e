import os
from typing import Dict, Any, List

class _Null:
    def emit(self, _: Dict[str, Any]): pass

class MetricsBus:
    def __init__(self):
        self.sinks: List[object] = []

    def register(self, sink): self.sinks.append(sink)
    def emit(self, payload: Dict[str, Any]):
        for s in self.sinks:
            try: s.emit(payload)
            except Exception as e: print("metrics sink error:", e)

metrics_bus = MetricsBus()

def init_from_env():
    if os.getenv("CFE_METRICS_PROM", "0") == "1":
        try:
            from .prom_exporter import PromExporter
            metrics_bus.register(PromExporter())
        except Exception as e:
            print("Prom exporter init failed:", e)
    if os.getenv("CFE_METRICS_OTEL", "0") == "1":
        try:
            from .otel_exporter import OTelExporter
            metrics_bus.register(OTelExporter())
        except Exception as e:
            print("OTel exporter init failed:", e)

# call init_from_env() once at process start (e.g., trainer entrypoint)
