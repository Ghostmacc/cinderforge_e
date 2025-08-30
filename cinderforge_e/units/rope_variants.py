# cinderforge_e/units/rope_variants.py
# RoPE + iRoPE helpers; assumes q,k: [B, H, T, Dh] with Dh even.
from typing import Tuple
import torch

@torch.no_grad()
def apply_rope(q: torch.Tensor, k: torch.Tensor, T: int, base: float = 10_000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:  # no-op if unexpected shape
        return q, k
    B, H, _, Dh = q.shape
    half = Dh // 2
    if half == 0:
        return q, k
    device, dtype = q.device, q.dtype
    freqs = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = base ** (-freqs.clamp(min=1) / max(half, 1))
    inv_freq = inv_freq.to(dtype)
    pos = torch.arange(T, device=device, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]  # [T, half]
    c = torch.cos(angles)[None, None, :, :].to(dtype)
    s = torch.sin(angles)[None, None, :, :].to(dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

    return rotate(q), rotate(k)

def _build_interleave_perm(Dh: int, groups: int, device, dtype=torch.long):
    groups = max(1, int(groups))
    if groups <= 1 or (Dh % groups) != 0:
        perm = torch.arange(Dh, device=device, dtype=dtype)
        return perm, perm
    per  = Dh // groups
    grid = torch.arange(Dh, device=device, dtype=dtype).view(groups, per)
    cols = []
    for c in range(per):
        for g in range(groups):
            cols.append(grid[g, c].item())
    perm = torch.tensor(cols, device=device, dtype=dtype)
    inv  = torch.empty_like(perm)
    inv[perm] = torch.arange(Dh, device=device, dtype=dtype)
    return perm, inv

@torch.no_grad()
def apply_irope(q: torch.Tensor, k: torch.Tensor, T: int, base: float = 10_000.0, groups: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        return q, k
    Dh = q.shape[-1]
    perm, inv = _build_interleave_perm(Dh, groups, q.device)
    qp = q.index_select(-1, perm)
    kp = k.index_select(-1, perm)
    qr, kr = apply_rope(qp, kp, T, base=base)
    return qr.index_select(-1, inv), kr.index_select(-1, inv)

def apply_posenc(q: torch.Tensor, k: torch.Tensor, T: int, variant: str = "rope", base: float = 10_000.0, groups: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    v = (variant or "rope").lower()
    if v == "irope":
        return apply_irope(q, k, T, base=base, groups=groups)
    return apply_rope(q, k, T, base=base)
