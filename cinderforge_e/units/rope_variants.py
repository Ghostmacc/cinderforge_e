# cinderforge_e/units/rope_variants.py
# RoPE + iRoPE helpers; assumes q,k: [B, H, T, Dh] with Dh even.
from typing import Tuple, Optional
import torch

# Global fixed projection matrix for stable witness tapping (initialized once)
_witness_projection = None

def get_witness_projection(Dh: int, device, dtype) -> torch.Tensor:
    """Get or create fixed orthonormal projection for witness tapping."""
    global _witness_projection
    if _witness_projection is None or _witness_projection.shape[0] != Dh:
        # Create fixed orthonormal basis for witness stability
        proj = torch.randn(Dh, min(Dh, 32), device=device, dtype=dtype)
        proj, _ = torch.qr(proj)  # QR decomposition for orthonormal basis
        _witness_projection = proj
    return _witness_projection.to(device=device, dtype=dtype)

def rescale_head_variance(x: torch.Tensor, target_std: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Rescale each head to maintain uniform variance after rotation.
    Args:
        x: tensor of shape [B, H, T, Dh]
        target_std: target standard deviation
        eps: small constant for numerical stability
    """
    B, H, T, Dh = x.shape
    # Compute per-head standard deviation
    head_std = x.view(B, H, -1).std(dim=-1, keepdim=True)  # [B, H, 1]
    head_std = head_std.clamp(min=eps)
    
    # Rescale each head
    scale_factor = target_std / head_std  # [B, H, 1]
    return x * scale_factor.unsqueeze(-1)  # [B, H, T, Dh]

@torch.no_grad()
def apply_rope(q: torch.Tensor, k: torch.Tensor, T: int, base: float = 10_000.0, head_scale_fix: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
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
        rotated = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        # Apply head variance rescaling if requested
        if head_scale_fix:
            rotated = rescale_head_variance(rotated)
        return rotated

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
def apply_irope(q: torch.Tensor, k: torch.Tensor, T: int, base: float = 10_000.0, groups: int = 2, head_scale_fix: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 4 or k.ndim != 4:
        return q, k
    Dh = q.shape[-1]
    perm, inv = _build_interleave_perm(Dh, groups, q.device)
    qp = q.index_select(-1, perm)
    kp = k.index_select(-1, perm)
    qr, kr = apply_rope(qp, kp, T, base=base, head_scale_fix=head_scale_fix)
    return qr.index_select(-1, inv), kr.index_select(-1, inv)

def apply_posenc(q: torch.Tensor, k: torch.Tensor, T: int, variant: str = "rope", base: float = 10_000.0, groups: int = 2, head_scale_fix: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    v = (variant or "rope").lower()
    if v == "irope":
        return apply_irope(q, k, T, base=base, groups=groups, head_scale_fix=head_scale_fix)
    return apply_rope(q, k, T, base=base, head_scale_fix=head_scale_fix)
