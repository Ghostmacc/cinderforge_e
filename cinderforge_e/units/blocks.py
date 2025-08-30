import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Dict
from .rope_variants import apply_posenc

# ---- Simple registry ----
UNIT_REGISTRY: Dict[str, Callable] = {}
def register(name: str):
    def _fn(fn: Callable):
        UNIT_REGISTRY[name] = fn
        return fn
    return _fn

# ---- MLP block (baseline) ----
@register("mlp_block")
def mlp_block(d_model: int, hidden: int | None = None, p: float = 0.1):
    h = hidden or 2 * d_model
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(d_model, h), nn.GELU(), nn.Dropout(p),
        nn.Linear(h, d_model),
    )

# ---- Multi‑Head Linear Attention + RoPE ----
class _MHLinearAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.h = n_heads
        self.d = d_model
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

        # iRoPE knobs (env-driven; defaults preserve old behavior)
        self.rope_variant = os.getenv("CFE_ROPE_VARIANT", "rope").lower()  # "rope" or "irope"
        self.rope_base    = float(os.getenv("CFE_ROPE_BASE", "10000"))
        self.irope_groups = int(os.getenv("CFE_IROPE_GROUPS", "2"))
        self.head_scale_fix = os.getenv("CFE_HEAD_SCALE_FIX", "true").lower() == "true"

    @staticmethod
    def _rope(q: torch.Tensor, k: torch.Tensor, base: float = 10000.0):
        # q,k: [B,H,T,Dh]
        B, H, T, Dh = q.shape
        half = Dh // 2
        if half == 0:
            return q, k
        device = q.device
        dtype = q.dtype
        freqs = torch.arange(half, device=device, dtype=dtype)
        inv_freq = base ** (-freqs / half)
        pos = torch.arange(T, device=device, dtype=dtype)
        angles = pos[:, None] * inv_freq[None, :]  # [T, half]
        c = torch.cos(angles)[None, None, :, :]    # [1,1,T,half]
        s = torch.sin(angles)[None, None, :, :]    # [1,1,T,half]

        def rotate(x):
            x1 = x[..., :half]
            x2 = x[..., half:2*half]
            rest = x[..., 2*half:]
            xr1 = x1 * c - x2 * s
            xr2 = x1 * s + x2 * c
            return torch.cat([xr1, xr2, rest], dim=-1)

        return rotate(q), rotate(k)

    @staticmethod
    def _phi(x: torch.Tensor):
        # Positive feature map for linear attention
        return F.elu(x, alpha=1.0) + 1.0

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).permute(0, 2, 1, 3)  # [B,H,T,Dh]
        k = self.k(x).view(B, T, self.h, self.dh).permute(0, 2, 1, 3)
        v = self.v(x).view(B, T, self.h, self.dh).permute(0, 2, 1, 3)

        # positional rotation (RoPE / iRoPE)
        T = q.shape[2]
        q, k = apply_posenc(q, k, T,
                            variant=self.rope_variant,
                            base=self.rope_base,
                            groups=self.irope_groups,
                            head_scale_fix=self.head_scale_fix)
        # Feature map
        q = self._phi(q)
        k = self._phi(k)

        # Causal linear attention (prefix sums)
        K_sum = x.new_zeros(B, self.h, self.dh)          # [B,H,Dh]
        KV = x.new_zeros(B, self.h, self.dh, self.dh)    # [B,H,Dh,Dh]
        outs = []
        for t in range(T):
            k_t = k[:, :, t, :]                          # [B,H,Dh]
            v_t = v[:, :, t, :]
            K_sum = K_sum + k_t
            KV = KV + torch.einsum("bhd,bhe->bhde", k_t, v_t)  # [B,H,Dh,Dh]

            q_t = q[:, :, t, :]                          # [B,H,Dh]
            num = torch.einsum("bhd,bhde->bhe", q_t, KV)       # [B,H,Dh]
            den = torch.einsum("bhd,bhd->bh", q_t, K_sum).clamp_min(1e-6)  # [B,H]
            y_t = num / den.unsqueeze(-1)                # [B,H,Dh]
            outs.append(y_t)

        Y = torch.stack(outs, dim=2)                     # [B,H,T,Dh]
        Y = Y.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # [B,T,D]
        Y = self.o(Y)
        return self.drop(Y)

@register("mhla_rope")
def mhla_rope(d_model: int):
    return nn.Sequential(nn.LayerNorm(d_model), _MHLinearAttn(d_model, n_heads=4, dropout=0.1))

# ---- Tiny depthwise SSM block (kept for completeness) ----
class DepthwiseSSM(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 64):
        super().__init__()
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, bias=False)
        nn.init.kaiming_uniform_(self.dw.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor):
        # x: [B,T,D]
        xt = x.transpose(1, 2)                # [B,D,T]
        xt = F.pad(xt, (self.dw.kernel_size[0] - 1, 0))  # causal pad
        y = self.dw(xt)                       # [B,D,T]
        return y.transpose(1, 2)

@register("ssm_diag")
def ssm_diag(d_model: int):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        DepthwiseSSM(d_model, kernel_size=64),
        nn.GELU(),
        nn.Linear(d_model, d_model),
    )
