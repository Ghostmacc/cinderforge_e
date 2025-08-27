from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.nas.metrics import pack_metrics
from cinderforge_e.dsl.spec import GraphSpec

def determinism_by_seed(spec: GraphSpec) -> bool:
    f1 = generate_futures(spec)
    f2 = generate_futures(spec)
    def edges(f):
        return sorted([(e.src.level, e.src.index, e.dst.level, e.dst.index, e.kind) for e in f.vertical + f.lateral])
    return edges(f1) == edges(f2)

def estimate_params(spec, fut):
    return sum(l.units * l.neurons_per * 2 for l in spec.levels)

def run_conformance(spec):
    fut = generate_futures(spec)
    det = determinism_by_seed(spec)
    met = pack_metrics(spec, fut)
    p = estimate_params(spec, fut)
    return {"determinism_by_seed": bool(det), "params_estimate": p, "structure": met, "ok": bool(det)}