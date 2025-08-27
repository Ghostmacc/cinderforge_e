import torch
import torch.nn as nn
from cinderforge_e.units.blocks import UNIT_REGISTRY

class ModelWrapper(nn.Module):
    """Returns raw logits (no sigmoid) so we can use BCEWithLogitsLoss safely with AMP."""
    def __init__(self, blocks, d_model: int):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x: torch.Tensor):
        # Accept [B,D] (tabular) or [B,T,D] (sequence)
        if x.ndim == 2:
            x = x.unsqueeze(1)  # [B,1,D]
        h = x
        for blk in self.blocks:
            h = h + blk(h)
        if h.ndim == 3:
            h = h.mean(dim=1)   # global average over time
        logits = self.head(h).squeeze(-1)  # [B]
        return logits

def build_model_from_spec(spec, fut):
    """Builds a simple stack of units determined by spec.resources['unit_type']."""
    d = spec.levels[0].neurons_per
    unit_name = (spec.resources or {}).get("unit_type", "mlp_block")
    unit = UNIT_REGISTRY.get(unit_name, UNIT_REGISTRY["mlp_block"])
    blocks = [unit(d) for _ in spec.levels]
    return ModelWrapper(blocks, d)
