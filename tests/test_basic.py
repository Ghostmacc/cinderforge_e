from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.nas.metrics import pack_metrics

def test_metrics_smoke():
    spec = GraphSpec(id="t", seed=1, levels=[LevelSpec(units=6, neurons_per=8)])
    fut = generate_futures(spec)
    m = pack_metrics(spec, fut)
    assert "n_nodes" in m and m["n_nodes"] >= 6