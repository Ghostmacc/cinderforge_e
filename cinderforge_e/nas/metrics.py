import math
import numpy as np
import networkx as nx
from typing import Dict, Optional
from cinderforge_e.dsl.spec import GraphSpec, Futures

def _graph(spec: GraphSpec, fut: Futures) -> nx.Graph:
    G = nx.DiGraph()
    for L, lvl in enumerate(spec.levels):
        for j in range(lvl.units):
            G.add_node((L, j))
    for e in fut.vertical + fut.lateral:
        G.add_edge((e.src.level, e.src.index), (e.dst.level, e.dst.index))
    return G.to_undirected()

def _largest_cc_avg_path_len(Gu: nx.Graph) -> Optional[float]:
    if Gu.number_of_nodes() < 2 or Gu.number_of_edges() == 0:
        return None
    H = Gu.subgraph(max(nx.connected_components(Gu), key=len)).copy()
    if H.number_of_nodes() < 2:
        return None
    try:
        return nx.average_shortest_path_length(H)
    except:
        return None

def small_world_sigma(Gu: nx.Graph) -> Optional[float]:
    N, E = Gu.number_of_nodes(), Gu.number_of_edges()
    if N < 3 or E == 0:
        return None
    C = nx.average_clustering(Gu)
    L = _largest_cc_avg_path_len(Gu)
    if L is None:
        return None
    k = (2.0 * E) / N
    if k <= 1.0:
        return None
    C_rand = k / N
    L_rand = math.log(N) / math.log(k)
    if C_rand <= 0 or L_rand <= 0:
        return None
    return float((C / C_rand) / (L / L_rand))

def degree_gini(Gu: nx.Graph) -> float:
    degs = np.array([d for _, d in Gu.degree()], float)
    if degs.size == 0:
        return 0.0
    mean = degs.mean()
    if mean == 0:
        return 0.0
    diffs = np.abs(degs[:, None] - degs[None, :])
    return float(diffs.sum() / (2.0 * (degs.size ** 2) * mean))

def hub_fraction(Gu: nx.Graph, top_p: float = 0.10) -> float:
    N, E = Gu.number_of_nodes(), Gu.number_of_edges()
    if N == 0 or E == 0:
        return 0.0
    top_n = max(1, int(math.ceil(top_p * N)))
    degs = sorted([d for _, d in Gu.degree()], reverse=True)
    return float(min(1.0, (sum(degs[:top_n]) / (2.0 * E))))

def spectral_radius(Gu: nx.Graph) -> float:
    if Gu.number_of_nodes() == 0:
        return 0.0
    A = nx.to_numpy_array(Gu, dtype=float)
    try:
        return float(np.max(np.abs(np.linalg.eigvals(A))).real)
    except:
        return 0.0

def pack_metrics(spec: GraphSpec, fut: Futures) -> Dict:
    Gu = _graph(spec, fut)
    sigma = small_world_sigma(Gu)
    C = float(nx.average_clustering(Gu)) if Gu.number_of_edges() > 0 else 0.0
    L = _largest_cc_avg_path_len(Gu)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        from networkx.algorithms.community.quality import modularity as nx_modularity
        coms = list(greedy_modularity_communities(Gu)) if Gu.number_of_edges() > 0 else [set(Gu.nodes)]
        Q = float(nx_modularity(Gu, coms)) if Gu.number_of_edges() > 0 else 0.0
    except:
        Q = 0.0
    return {
        "n_nodes": Gu.number_of_nodes(),
        "n_edges": Gu.number_of_edges(),
        "avg_clustering": C,
        "avg_path_len": float(L) if L is not None else None,
        "sigma_overall": sigma,
        "modularity": Q,
        "degree_gini": float(degree_gini(Gu)),
        "hub_fraction": float(hub_fraction(Gu)),
        "spectral_radius": float(spectral_radius(Gu)),
    }