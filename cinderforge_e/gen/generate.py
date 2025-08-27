import random
import networkx as nx
from cinderforge_e.dsl.spec import GraphSpec, UnitAddr, Futures, Edge
from cinderforge_e.gen.topology import lateral_small_world, lateral_scale_free

def generate_futures(spec: GraphSpec) -> Futures:
    rng = random.Random(spec.seed)
    fut = Futures()
    levels = [[UnitAddr(level=i, index=j) for j in range(spec.levels[i].units)] for i in range(len(spec.levels))]
    for L in range(1, len(levels)):
        for u in levels[L]:
            src = [UnitAddr(level=k, index=j) for k in range(max(0, L - spec.max_vertical_skip), L) for j in range(len(levels[k]))]
            if src:
                fut.vertical.append(Edge(src=rng.choice(src), dst=u, kind="vertical"))
    sel = {(e.src.level, e.src.index) for e in fut.vertical}
    for L in range(0, len(levels) - 1):
        for u in levels[L]:
            if (u.level, u.index) not in sel:
                dst = [UnitAddr(level=k, index=j) for k in range(L + 1, min(len(levels), L + 1 + spec.max_vertical_skip)) for j in range(len(levels[k]))]
                if dst:
                    fut.vertical.append(Edge(src=u, dst=rng.choice(dst), kind="vertical"))
    mode = (spec.resources or {}).get("lateral_mode", "affinity")
    if mode in ("ws", "small_world"):
        k = int((spec.resources or {}).get("ws_k", max(1, spec.max_lateral_right)))
        beta = float((spec.resources or {}).get("ws_beta", 0.15))
        for L in range(len(levels)):
            fut.lateral.extend(lateral_small_world(levels[L], beta=beta, k=k, rng=rng))
    elif mode in ("ba", "scale_free"):
        m = int((spec.resources or {}).get("ba_m", 1))
        for L in range(len(levels)):
            fut.lateral.extend(lateral_scale_free(levels[L], m=m, rng=rng))
    else:
        for L in range(len(levels)):
            for j, u in enumerate(levels[L]):
                cap = min(spec.max_lateral_right, max(0, len(levels[L]) - (j + 1)))
                for step in range(1, cap + 1):
                    fut.lateral.append(Edge(src=u, dst=levels[L][j + step], kind="lateral"))
    G = nx.DiGraph()
    for e in fut.vertical + fut.lateral:
        G.add_edge((e.src.level, e.src.index), (e.dst.level, e.dst.index))
    assert nx.is_directed_acyclic_graph(G), "generator produced a cycle"
    return fut