# cinderforge_e/trainer/step.py
import time
import json
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch

from ..dynamics.aggregator import Aggregator
from ..dynamics.penalty import dynamics_penalty
from ..dynamics.autoreg import AutoReg


@dataclass
class TrainingWeights:
    """Mutable weights for dynamics penalty."""
    plv: float = 0.75
    sigma: float = 0.50
    lambda_: float = 1.00


class MockWitnesses:
    """Mock witness system for PLV, sigma, and lambda calculations."""
    
    def plv(self, aux: Dict[str, torch.Tensor]) -> tuple[float, float]:
        """Compute Phase Locking Value and Rayleigh p-value from attention data."""
        # Mock implementation - in real system this would compute from attention matrices
        if 'attn_weights' in aux:
            attn = aux['attn_weights']  # [B, H, T, T] or similar
            # Simple mock: use attention entropy as proxy for coherence
            entropy = -(attn * torch.log(attn + 1e-8)).sum(dim=-1).mean()
            plv = torch.clamp(1.0 - entropy / 10.0, 0.0, 1.0).item()
            p_rayleigh = 0.1 * torch.rand(1).item()  # Mock Rayleigh test p-value
        else:
            plv = 0.5 + 0.1 * torch.randn(1).item()
            p_rayleigh = 0.1 * torch.rand(1).item()
        return plv, p_rayleigh
    
    def branching(self, aux: Dict[str, torch.Tensor]) -> float:
        """Compute branching parameter sigma."""
        # Mock implementation
        return 1.0 + 0.1 * torch.randn(1).item()
    
    def local_lyapunov(self, aux: Dict[str, torch.Tensor]) -> float:
        """Compute local Lyapunov exponent."""
        # Mock implementation - should be <= 0 for stability
        return -0.05 + 0.1 * torch.randn(1).item()


def train_step(batch, state, cfg, witnesses: MockWitnesses, agg: Aggregator, 
               auto: AutoReg, weights: TrainingWeights, step_idx: int, 
               steps_per_epoch: int = 100) -> float:
    """
    Integrated training step with dynamics monitoring and control.
    
    Args:
        batch: Training batch data
        state: Training state (model, optimizer, etc.)
        cfg: Configuration object from cfe_config.yaml
        witnesses: Witness computation system
        agg: Verdict aggregator with Schmitt trigger
        auto: Auto-regulation system
        weights: Current penalty weights
        step_idx: Current global step
        steps_per_epoch: Steps per epoch for auto-correct timing
        
    Returns:
        Total loss value
    """
    # Forward pass with attention taps
    if hasattr(state.model, 'return_attn'):
        logits, aux = state.model(batch, return_attn=True)
    else:
        # Fallback for models without attention tapping
        logits = state.model(batch)
        aux = {}
    
    # 1) Witness sampling from taps (e.g., head-averaged pre-softmax phases)
    plv, p_ray = witnesses.plv(aux)
    sigma = witnesses.branching(aux)
    lam = witnesses.local_lyapunov(aux)
    
    # 2) Gate verdict (for logging + rhythm, if you stagger edits)
    verdict = agg.fuse(plv, sigma, lam, p_ray)
    
    # 3) Base loss
    task_loss = state.loss_fn(logits, batch.get('y', batch.get('target')))
    
    warmup_left = max(0, cfg.irope.warmup_steps - step_idx)
    dyn_loss, err = dynamics_penalty(
        {'plv': plv, 'sigma': sigma, 'lambda': lam},
        cfg.targets, weights, cfg.gate.mode, warmup_left
    )
    total_loss = task_loss + dyn_loss
    
    # 4) Backprop / step
    state.opt.zero_grad()
    total_loss.backward()
    state.opt.step()
    
    # 5) Auto-regulator (epoch boundary or cadence)
    if (cfg.gate.mode == "auto" and cfg.auto.enabled and 
        step_idx % steps_per_epoch == 0 and step_idx > 0):
        weights, err_auto = auto.update(
            {'plv': plv, 'sigma': sigma, 'lambda': lam}, cfg.targets, weights
        )
    else:
        err_auto = {}
    
    # 6) Log receipts (tokens + heartbeat) — keep your JSON schema stable
    if hasattr(state, 'logger'):
        # Log tokens if token logging is enabled
        if hasattr(state.logger, 'tokens'):
            state.logger.tokens(batch, logits)
        
        # Log heartbeat with full schema
        heartbeat_data = {
            "t": int(time.time()),
            "step": step_idx,
            "task_loss": float(task_loss.item()),
            "dyn_loss": float(dyn_loss) if isinstance(dyn_loss, torch.Tensor) else float(dyn_loss),
            "total": float(total_loss.item()),
            "plv": float(plv),
            "sigma": float(sigma),
            "lambda_": float(lam),
            "p": float(p_ray),
            "gate_open": verdict.open_,
            "gate_score": float(verdict.score),
            "mode": cfg.gate.mode,
            "weights": {
                "plv": weights.plv,
                "sigma": weights.sigma,
                "lambda_": weights.lambda_
            },
            "err": err,
            "err_auto": err_auto,
            "irope": {
                "enabled": cfg.irope.enabled,
                "interleave": cfg.irope.interleave_every,
                "warmup_left": warmup_left
            }
        }
        
        if hasattr(state.logger, 'heartbeat'):
            state.logger.heartbeat(heartbeat_data)
        else:
            # Fallback: write to heartbeat file directly
            import json
            from pathlib import Path
            hb_path = getattr(state, 'heartbeat_path', 'heartbeat.ndjson')
            Path(hb_path).parent.mkdir(parents=True, exist_ok=True)
            with open(hb_path, 'a') as f:
                f.write(json.dumps(heartbeat_data) + '\n')
    
    return float(total_loss.item())


def create_training_components(cfg):
    """
    Factory function to create training components from config.
    
    Args:
        cfg: Configuration object from cfe_config.yaml
        
    Returns:
        tuple: (witnesses, aggregator, auto_reg, weights)
    """
    witnesses = MockWitnesses()
    agg = Aggregator(cfg)
    auto = AutoReg(cfg)
    weights = TrainingWeights(
        plv=cfg.weights.plv,
        sigma=cfg.weights.sigma,
        lambda_=cfg.weights.lambda_
    )
    
    return witnesses, agg, auto, weights
