from prometheus_client import start_http_server, Gauge
import os, time

class PromExporter:
    def __init__(self):
        port = int(os.getenv("CFE_PROM_PORT", "9808"))
        start_http_server(port)
        # Gauges we care about from heartbeat
        self.g_task = Gauge("cfe_task_loss", "Task loss")
        self.g_dyn  = Gauge("cfe_dyn_loss", "Dynamics penalty loss")
        self.g_plv  = Gauge("cfe_plv", "Phase-locking value")
        self.g_sig  = Gauge("cfe_sigma", "Branching factor")
        self.g_lam  = Gauge("cfe_lambda", "Local Lyapunov proxy")
        self.g_gate = Gauge("cfe_gate_score", "Gate fused score")
        self.g_mode = Gauge("cfe_mode", "Mode (0=off,1=observe,2=gentle,3=hard,4=auto)")
        self.g_weight_plv = Gauge("cfe_w_plv", "Weight plv")
        self.g_weight_sig = Gauge("cfe_w_sigma", "Weight sigma")
        self.g_weight_lam = Gauge("cfe_w_lambda", "Weight lambda")

    def _mode_to_num(self, m:str)->int:
        table = {"off":0,"observe":1,"gentle":2,"hard":3,"auto":4}
        return table.get(m, -1)

    def emit(self, hb):
        self.g_task.set(hb.get("task_loss", 0.0))
        self.g_dyn.set(hb.get("dyn_loss", 0.0))
        self.g_plv.set(hb.get("plv", 0.0))
        self.g_sig.set(hb.get("sigma", 0.0))
        self.g_lam.set(hb.get("lambda_", 0.0))
        self.g_gate.set(hb.get("gate_score", 0.0))
        self.g_mode.set(self._mode_to_num(hb.get("mode","observe")))
        w = hb.get("weights", {})
        self.g_weight_plv.set(w.get("plv", 0.0))
        self.g_weight_sig.set(w.get("sigma", 0.0))
        self.g_weight_lam.set(w.get("lambda_", 0.0))
