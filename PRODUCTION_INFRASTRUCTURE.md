# Production Infrastructure Guide

This document outlines the complete production-ready infrastructure for Cinderforge-E, including distributed training, metrics exporters, and scaling guidelines.

## 🏗️ Architecture Overview

The production infrastructure provides:

- **Config Overlays**: Environment-specific YAML configurations
- **DeepSpeed ZeRO-3**: Memory-efficient distributed training
- **Ray Train Launcher**: Scalable cluster training with torchrun semantics
- **Prometheus + OTEL**: Real-time metrics and observability
- **Distributed-Safe Aggregation**: Consistent PLV/σ/λ computation across ranks
- **Edge-of-Chaos Guardrails**: Scaling safeguards and failure mode protection

## 📁 File Structure

```
cinderforge_e/
├── cluster/
│   ├── deepspeed/
│   │   └── ds_config_zero3.json          # DeepSpeed ZeRO-3 configuration
│   └── ray/
│       └── cluster_ray.yaml              # Ray cluster configuration
├── cinderforge_e/
│   ├── launch/
│   │   ├── deepspeed_helpers.py           # DeepSpeed initialization helpers
│   │   └── ray_launcher.py                # Ray Train launcher
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── exporters.py                   # Prometheus/OTEL exporters
│   └── dynamics/
│       └── ddp_reduce.py                  # Distributed aggregation utilities
```

## 🚀 Run Recipes

### 1. Single GPU (Development/Sanity)

```bash
# Basic iRoPE + observe mode
CFE_ROPE_VARIANT=irope CFE_GATE_MODE=observe \
python -m cinderforge_e.trainer.train_seq --config cfe_config.yaml

# With Prometheus metrics
CFE_PROM_ENABLED=true CFE_PROM_PORT=9108 \
CFE_ROPE_VARIANT=irope CFE_GATE_MODE=auto \
python -m cinderforge_e.trainer.train_seq --config cfe_config.yaml
```

### 2. DeepSpeed ZeRO-3 (8×80GB GPUs)

```bash
# Environment setup
export CFE_LAUNCHER=deepspeed
export CFE_DS_CONFIG=cluster/deepspeed/ds_config_zero3.json
export CFE_GATE_MODE=auto         # off|observe|gentle|hard|auto
export CFE_PROM_ENABLED=true
export CFE_PROM_PORT=9108
export CFE_ROPE_VARIANT=irope
export CFE_IROPE_GROUPS=2

# Launch distributed training
deepspeed --num_gpus=8 --module cinderforge_e.trainer.train_seq \
  --deepspeed $CFE_DS_CONFIG --config cfe_config.yaml
```

### 3. Ray Train (Multi-node Cluster)

```bash
# Start Ray head node
ray start --head

# Launch distributed training
python -m cinderforge_e.launch.ray_launcher \
  --num-workers 8 \
  --use-gpu \
  --nudge auto \
  --results ray_results/cfe

# Stop Ray when done
ray stop
```

## 📊 Metrics & Observability

### Prometheus Metrics

Enable Prometheus metrics export:

```bash
export CFE_PROM_ENABLED=true
export CFE_PROM_PORT=9108
```

Available metrics:
- `cfe_plv`: Phase locking value
- `cfe_sigma`: Branching factor (criticality)
- `cfe_lambda`: Local Lyapunov proxy
- `cfe_gate_score`: Fused gate score [0,1]
- `cfe_dyn_loss`: Dynamics penalty loss
- `cfe_task_loss`: Task loss
- `cfe_gate_open`: Gate open (0/1)
- `cfe_p_rayleigh`: Rayleigh p-value
- `cfe_nudges_applied`: Auto-reg nudge counter
- `cfe_weight_*`: Dynamics weights (plv, sigma, lambda)

Scrape endpoint: `http://localhost:9108/metrics`

### OpenTelemetry (OTEL)

Enable OTEL export:

```bash
export CFE_OTEL_ENABLED=true
export CFE_OTEL_ENDPOINT=http://otel-collector:4318/v1/metrics
```

## ⚖️ Distributed Training Integration

### DeepSpeed Integration Pattern

Your existing trainer can be enhanced with DeepSpeed support:

```python
# In your train_seq.py or equivalent
import os
USE_DEEPSPEED = os.getenv("CFE_LAUNCHER", "").lower() == "deepspeed"

if USE_DEEPSPEED:
    import deepspeed
    from cinderforge_e.launch.deepspeed_helpers import maybe_init_dist, is_main, load_ds_config
    
    maybe_init_dist()
    ds_cfg = load_ds_config(os.getenv("CFE_DS_CONFIG", "cluster/deepspeed/ds_config_zero3.json"))
    
    model = build_model(cfg).cuda()
    optimizer = build_optimizer(model, cfg)
    
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_cfg
    )
else:
    # Normal single-GPU/torchrun path
    model = build_model(cfg).cuda()
    optimizer = build_optimizer(model, cfg)

# In training loop
if USE_DEEPSPEED:
    model.backward(total_loss)
    model.step()
else:
    total_loss.backward()
    optimizer.step()
```

### Distributed-Safe Dynamics

Ensure consistent PLV/σ/λ computation across ranks:

```python
from cinderforge_e.dynamics.ddp_reduce import allreduce_mean, broadcast_weights

# Compute per-rank metrics, then reduce
plv_local = compute_plv(...)
sigma_local = compute_sigma(...)
lambda_local = compute_lambda(...)

# All-reduce to get consistent metrics
plv = allreduce_mean(plv_local)
sigma = allreduce_mean(sigma_local)
lam = allreduce_mean(lambda_local)

# Compute verdict and auto-reg updates on rank 0, then broadcast
if cfg.gate.mode == "auto" and is_main_rank():
    weights, err_auto = auto.update(
        {"plv": plv, "sigma": sigma, "lambda_": lam}, 
        cfg.targets, 
        weights
    )
weights = broadcast_weights(weights)
```

### Metrics Export Integration

Add to your heartbeat logging:

```python
# After composing hb_dict for heartbeat.ndjson
from cinderforge_e.metrics.exporters import MetricsExport

if not hasattr(state, "_metrics_export"):
    state._metrics_export = MetricsExport(
        port=int(os.getenv("CFE_PROM_PORT", "9108"))
    )
state._metrics_export.handle_heartbeat(hb_dict)
```

## 📈 Scaling Guidelines & Guardrails

### Edge-of-Chaos at Scale

**Sampling Rate**: Compute PLV/σ/λ on sliding subwindows (1-2 heads × 128 tokens per step) to keep O(T) cost manageable.

**Fixed Witness Projection**: Use identical orthonormal projection across ranks (same seed on rank 0, broadcast).

**Numerics Stability**: BF16 + ZeRO-3 can alter gradient noise. Schmitt hysteresis prevents spurious gate flips:
- Open threshold: 0.75
- Close threshold: 0.65  
- `min_inter_gate_steps` for stability

**Auto-Reg Bounds**: 
- Keep `max_step ≤ 0.1`
- Add epoch cooldown
- Critical when scaling world size to avoid ping-pong

### Failure Modes & Prevention

**Rank Drift**: Without distributed aggregation, different ranks may make conflicting decisions. Solution: `allreduce_mean()` for metrics, `broadcast_weights()` for auto-reg.

**Gate Chatter**: High-frequency open/close cycles under distributed noise. Solution: Schmitt hysteresis + minimum inter-gate steps.

**Memory Explosion**: Large models with full attention. Solution: DeepSpeed ZeRO-3 with CPU offload.

**Gradient Noise**: BF16 mixed precision affects dynamics stability. Solution: Proper thresholds and bounded PI control.

## 🔧 Configuration Reference

### DeepSpeed Configuration

`cluster/deepspeed/ds_config_zero3.json` provides:
- BF16 mixed precision
- ZeRO-3 parameter/optimizer partitioning
- CPU offloading for memory efficiency
- Gradient clipping and communication overlap

For 8×80GB GPUs, you can disable offload and increase bucket sizes:

```json
{
  "offload_param": { "device": "none" },
  "offload_optimizer": { "device": "none" },
  "reduce_bucket_size": 200000000,
  "stage3_prefetch_bucket_size": 200000000
}
```

### Environment Variables

Core settings:
- `CFE_LAUNCHER`: `single|deepspeed|torchrun` 
- `CFE_GATE_MODE`: `off|observe|gentle|hard|auto`
- `CFE_ROPE_VARIANT`: `rope|irope`
- `CFE_IROPE_GROUPS`: Number of interleaving groups (default: 2)

DeepSpeed:
- `CFE_DS_CONFIG`: Path to DeepSpeed JSON config

Metrics:
- `CFE_PROM_ENABLED`: Enable Prometheus export
- `CFE_PROM_PORT`: Prometheus server port (default: 9108)
- `CFE_OTEL_ENABLED`: Enable OpenTelemetry export
- `CFE_OTEL_ENDPOINT`: OTEL collector endpoint

## 🎯 Production Checklist

- [ ] **Config Overlays**: Environment-specific YAML configurations
- [ ] **DeepSpeed**: ZeRO-3 configuration tuned for your hardware
- [ ] **Distributed Aggregation**: PLV/σ/λ reduction implemented
- [ ] **Metrics Export**: Prometheus/OTEL integrated with heartbeat
- [ ] **Failure Recovery**: Checkpointing with ZeRO state saving
- [ ] **Monitoring**: Grafana dashboards for dynamics metrics
- [ ] **Scaling Tests**: Validated on target cluster configuration

## 📚 Dependencies

Additional packages required for production features:

```bash
pip install deepspeed>=0.10.0          # For ZeRO-3 training
pip install ray[train]>=2.34           # For distributed Ray training  
pip install prometheus_client>=0.15.0  # For Prometheus metrics
pip install opentelemetry-api          # For OTEL metrics (optional)
```

## 🛠️ Troubleshooting

**Import Errors**: The optional dependencies (Ray, DeepSpeed, Prometheus, OTEL) are only imported when their respective environment flags are enabled.

**CUDA OOM**: Increase DeepSpeed offload settings or reduce `train_micro_batch_size_per_gpu`.

**Gate Instability**: Check Schmitt thresholds and `min_inter_gate_steps` in your config.

**Metrics Missing**: Verify `CFE_PROM_ENABLED=true` and that the heartbeat integration is called after each step.

**Rank Inconsistency**: Ensure distributed aggregation functions are called consistently across all ranks.

---

This infrastructure enables seamless scaling from single-GPU development to multi-node production clusters while maintaining the edge-of-chaos dynamics that make Cinderforge-E effective.
