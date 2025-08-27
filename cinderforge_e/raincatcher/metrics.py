import functools
import random
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import hilbert
import networkx as nx

def _hash_embed(tokens, dim: int = 256) -> torch.Tensor:
    """Fast, dependency-free fallback embedding."""
    arr = np.empty((len(tokens), dim), dtype=np.float32)
    for i, t in enumerate(tokens):
        h = abs(hash(t))
        arr[i] = np.array([1.0 if (h >> (b % 32)) & 1 else -1.0 for b in range(dim)], dtype=np.float32)
    return torch.from_numpy(arr)

@functools.lru_cache(maxsize=1)
def _load_st(model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name, device=device)

def embed_tokens(tokens, device: str = "cpu", precision: str = "fp32",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 prefer_st: bool = True) -> torch.Tensor:
    """Return [N,D] embeddings. Uses SentenceTransformers if available, else hashed fallback."""
    if prefer_st:
        try:
            st = _load_st(model_name, device)
            embs = st.encode(tokens, convert_to_numpy=True, device=device, normalize_embeddings=True)
            return torch.from_numpy(np.asarray(embs))
        except Exception:
            pass
    return _hash_embed(tokens)

def compute_plv_from_embeddings(embeddings, window: int = 128) -> float:
    """PLV from pairwise cosine-similarity time-series of embeddings."""
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if not torch.is_tensor(embeddings) or embeddings.ndim != 2 or embeddings.shape[0] < 2:
        return 0.0
    sims = [F.cosine_similarity(embeddings[i], embeddings[i-1], dim=0).item()
            for i in range(1, embeddings.shape[0])]
    x = np.asarray(sims, dtype=np.float64)
    if x.size < 4:
        return 0.0
    analytic = hilbert(x)
    phases = np.angle(analytic)
    win = phases[-min(window, phases.size):]
    return float(np.abs(np.mean(np.exp(1j * win))))

def lyapunov_from_embeddings(embeddings, eps: float = 1e-6) -> float:
    """Local Lyapunov proxy from second-difference growth on cosine-sim series."""
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(embeddings)
    if not torch.is_tensor(embeddings) or embeddings.shape[0] < 3:
        return 0.0
    sims = [F.cosine_similarity(embeddings[i], embeddings[i-1], dim=0).item()
            for i in range(1, embeddings.shape[0])]
    x = np.asarray(sims, dtype=np.float64)
    if x.size < 3:
        return 0.0
    dx = np.diff(x)
    prev = np.maximum(np.abs(dx[:-1]), eps)
    ratio = np.abs(dx[1:]) / prev
    return float(np.mean(np.log(ratio + eps)))

def branching_factor(tokens) -> float:
    """σ proxy = average out-degree over the symbol transition graph."""
    if not tokens or len(tokens) < 2:
        return 0.0
    G = nx.DiGraph()
    for i in range(len(tokens) - 1):
        G.add_edge(tokens[i], tokens[i+1])
    degs = [d for _, d in G.out_degree()]
    return float(sum(degs) / len(degs)) if degs else 0.0

def add_noise(tokens, noise_level: float = 0.0):
    if noise_level <= 0:
        return tokens
    out = []
    for t in tokens:
        out.append(t if random.random() > noise_level else "[NOISE]")
    return out
