import os, time
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

class OTelExporter:
    def __init__(self):
        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4317")
        resource = Resource.create({"service.name": "cinderforge-e"})
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=endpoint))
        self.provider = MeterProvider(resource=resource, metric_readers=[reader])
        self.meter = self.provider.get_meter("cfe")
        # metrics
        self.task = self.meter.create_gauge("cfe.task_loss")
        self.dyn  = self.meter.create_gauge("cfe.dyn_loss")
        self.plv  = self.meter.create_gauge("cfe.plv")
        self.sigma= self.meter.create_gauge("cfe.sigma")
        self.lam  = self.meter.create_gauge("cfe.lambda")
        self.gate = self.meter.create_gauge("cfe.gate_score")

    def emit(self, hb):
        attrs = {"mode": hb.get("mode","observe")}
        self.task.record(hb.get("task_loss",0.0), attrs)
        self.dyn.record(hb.get("dyn_loss",0.0), attrs)
        self.plv.record(hb.get("plv",0.0), attrs)
        self.sigma.record(hb.get("sigma",0.0), attrs)
        self.lam.record(hb.get("lambda_",0.0), attrs)
        self.gate.record(hb.get("gate_score",0.0), attrs)
