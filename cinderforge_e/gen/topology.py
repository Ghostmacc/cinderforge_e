from random import Random
from typing import List
from cinderforge_e.dsl.spec import UnitAddr, Edge

def lateral_small_world(nodes: List[UnitAddr], beta: float, k: int, rng: Random):
    edges = []
    n = len(nodes)
    if n <= 1 or k <= 0:
        return edges
    for j in range(n):
        for step in range(1, min(k, n - (j + 1)) + 1):
            dst = j + step
            if rng.random() < float(beta):
                dst = rng.randrange(j + 1, n)
            edges.append(Edge(src=nodes[j], dst=nodes[dst], kind="lateral"))
    return edges

def lateral_scale_free(nodes: List[UnitAddr], m: int, rng: Random):
    edges = []
    n = len(nodes)
    if n <= 1 or m <= 0:
        return edges
    indeg = [0] * n
    for j in range(n - 1):
        cands = list(range(j + 1, n))
        chosen = []
        for _ in range(min(m, len(cands))):
            weights = [(indeg[idx] + 1) for idx in cands]
            tot = sum(weights)
            r = rng.random() * tot
            acc = 0.0
            pick = None
            for idx, w in zip(cands, weights):
                acc += w
                if acc >= r:
                    pick = idx
                    break
            if pick is None:
                pick = cands[-1]
            chosen.append(pick)
            cands.remove(pick)
            indeg[pick] += 1
        for dst in chosen:
            edges.append(Edge(src=nodes[j], dst=nodes[dst], kind="lateral"))
    return edges