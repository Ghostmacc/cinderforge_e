# CinderForge E
**Emergence‑Optimized Neural Architecture Search with Chaos‑Aware Instrumentation**

**Version:** 0.9 (engineering draft)  
**Date:** 2025‑08‑26

---

## Abstract

CinderForge E is a neural‑architecture search (NAS) system that explicitly **optimizes for emergent dynamics**, not only task loss. It couples a conventional NAS loop with **Raincatcher**, a lightweight, real‑time instrumentation stack that measures three dynamical signatures over token streams generated during training:

- **PLV (phase‑locking value)** — a measure of attractor occupancy/coherence,
- **σ (branching factor)** — a proxy for criticality (average out‑degree of a token transition graph),
- **λ (local Lyapunov proxy)** — a sign‑of‑expansion indicator derived from second‑difference growth on an embedding‑similarity series.

CinderForge E uses these metrics in three **regulation modes**: **observe** (measurement only), **gentle** (light nudging toward a target band), and **hard** (stronger nudging). A planned **auto** mode will adapt the nudging per epoch based on the recent dynamics. The result is a search process that **selects architectures for the right regime**—coherent yet adaptable (“edge‑of‑chaos”)—and produces **auditable receipts of loop health** alongside task metrics.

Our engineering goal is to create small, efficient models that **behave like larger ones** (e.g., customer‑service agents) by preserving useful emergent properties while avoiding runaway loops or sterile catatonia. CinderForge E currently integrates MHLA+RoPE, SSM blocks, linear attention, and MLP units; supports CUDA/BF16; and scales to long sequences via accumulation and capped optimizer steps while keeping O(n) inference for linear‑attention paths.

---

## 1. Motivation

Traditional NAS optimizes an objective like validation loss or accuracy. This ignores **how** a model produces its outputs—the **dynamical structure** of the computation. Empirical evidence from recurrent/attention systems suggests that useful behavior often arises near an “**edge‑of‑chaos**” regime: too much order → brittleness/catatonia; too much chaos → thrash/hallucination. Standard objectives provide no handle to locate and preserve this regime.

**CinderForge E** makes the regime **measurable and steerable**. By logging a coarse token stream during training and passing it through **Raincatcher**, we obtain PLV/σ/λ signals that summarize attractor strength, branching/criticality, and local expansion. We then (optionally) nudge search toward a sweet spot while preserving novelty. This is *emergence optimization*, not mere loss minimization.

---

## 2. System Overview

### 2.1 High‑Level Diagram

```
+-------------------+       +---------------------+       +-------------------+
|  NAS Sampler      |       |  Trainer            |       |  Raincatcher      |
|  (Optuna)         |  -->  |  (CUDA/BF16)        |  -->  |  (PLV, σ, λ)      |
|  + GraphSpec DSL  |       |  emits tokens.ndjson|       |  JSON/MD receipts |
+-------------------+       +---------------------+       +-------------------+
           ^                           |                              |
           |                           v                              |
           |                    heartbeat.ndjson                      |
           |                                                         v
           +------------------  Objective  <--- chaos regularizer ----+
```

### 2.2 Key Components

- **GraphSpec/LevelSpec DSL** — hierarchical levels with tunable units per level and neurons per unit; vertical skips; capped lateral connections.
- **Unit types** — `mlp_block`, `linear_attn`, `ssm_diag`, `mhla_rope` (multi‑head linear attention with RoPE).
- **Lateral wiring modes** — `affinity`, `ws` (Watts‑Strogatz‑like small‑world), `ba` (Barabási–Albert‑like preferential attachment). These shape **σ_struct** and **modularity**.
- **Trainer** — BF16/FP16/FP32; supports `steps_per_epoch` and `accum_steps`; writes **`tokens.ndjson`** (coarse POS/NEG tokens per optimizer step) and **`heartbeat.ndjson`** (loss over time).
- **Raincatcher CLI** — ingests NDJSON or plaintext token streams, encodes tokens (SentenceTransformers or hashed fallback), and computes **PLV**, **σ**, **λ**. Emits JSON & Markdown summaries.
- **NAS Objective** — combines task loss with a **chaos regularizer** (optional) that nudges PLV/σ/λ toward target bands. Modes: **observe**, **gentle**, **hard** (planned **auto** per‑epoch switching).

---

## 3. Raincatcher Metrics

Let \( t_i \) denote tokens (e.g., “POS”, “NEG”) logged at optimizer steps; \( e_i \in \mathbb{R}^d \) are token embeddings.

### 3.1 PLV (Phase‑Locking Value)

1. Compute the time series of consecutive embedding cosines:
\[
s_i = \cos(e_i, e_{i-1}) \quad \text{for } i = 1..N-1.
\]
2. Obtain the analytic signal \(a = \text{Hilbert}(s)\) and its phase \( \phi = \angle a \).
3. For a window \(W\) (\(W \le N-1\)), define
\[
\mathrm{PLV} = \left| \frac{1}{W} \sum_{k=N-W}^{N-1} e^{j\phi_k} \right| \in [0,1].
\]

**Interpretation.** High PLV ⇒ strong phase alignment (attractor occupancy). Very high PLV (>0.8) risks catatonia; very low PLV ⇒ noise/aimlessness. **Sweet spot** ≈ 0.55–0.70 for “coherent but pliable” dynamics.

### 3.2 σ (Branching Factor)

Construct a directed multigraph over the token alphabet with an edge \( t_{i-1} \to t_i \) per transition. Define
\[
\sigma = \frac{1}{|V|} \sum_{v \in V} \mathrm{outdeg}(v).
\]
σ ≈ 1 indicates **critical branching** (edge‑of‑chaos); σ ≪ 1 ⇒ rigidity; σ ≫ 1 ⇒ thrash.

### 3.3 λ (Local Lyapunov Proxy)

Using the cosine series \( s \), compute first differences \( \Delta s \) and measure second‑difference growth as a **sign‑of‑expansion** proxy:
\[
\lambda = \mathbb{E}\left[ \log \frac{|\Delta s_{k+1}|}{\max(|\Delta s_k|, \varepsilon)} \right].
\]
We **penalize** \( \lambda > 0 \) (expansive/divergent); \( \lambda \le 0 \) is neutral/contractive.

### 3.4 Targets and Bands

- **PLV target** \( \mu \) with band \( \delta \) (default \( \mu=0.60, \delta=0.10 \)).
- **σ band** ±0.10 around 1.0.
- **λ penalty** only for \( \lambda > 0 \).

These deliver a principled **regularizer** that rewards edge‑of‑chaos without collapsing diversity.

---

## 4. NAS Objective

For trial \( \tau \) with final task loss \( L_\tau \) and Raincatcher metrics \( \mathrm{PLV}_\tau, \sigma_\tau, \lambda_\tau \), define a soft chaos penalty:
\[
\mathrm{dev}_{\mathrm{PLV}} = \max\left(0, |\mathrm{PLV}_\tau - \mu| - \delta \right), \quad
\mathrm{dev}_\sigma = \max\left(0, |\sigma_\tau - 1| - b_\sigma \right), \quad
\mathrm{pen}_\lambda = \max(0, \lambda_\tau).
\]
\[
\mathrm{ChaosPen}_\tau = w \cdot \left( 0.5\,\mathrm{dev}_{\mathrm{PLV}} + 0.5\,\mathrm{dev}_\sigma + 1.0\,\mathrm{pen}_\lambda \right),
\]
where \( w \) is the **mode weight** (observe=0, gentle≈0.02, hard≈0.08 by default). The final NAS score is
\[
J_\tau = L_\tau + \mathrm{ChaosPen}_\tau - c_1 \cdot \max(0, \sigma_{\mathrm{struct}} - 1) - c_2 \cdot \mathrm{modularity}.
\]

This **does not** hard‑constrain dynamics; it **nudges** trials that wander far from the target regime while preserving exploration and structural quality.

---

## 5. Algorithms (pseudo‑code)

### 5.1 CinderForge E (per trial)

```python
spec = sample_graphspec(trial)              # units per level, neurons, skips, lateral mode
fut  = materialize(spec)
model = build_model(spec, fut)

# Train with gradient accumulation and step caps (BF16 on CUDA if available)
tokens = []
for epoch in range(E):
    zero_grad()
    steps = 0
    for x, y in dataloader:
        logits = model(x)
        loss = BCEWithLogits(logits, y) / accum_steps
        loss.backward()
        if step % accum_steps == 0:
            optimizer.step(); optimizer.zero_grad()
            tokens.append("POS" if sigmoid(logits).mean() >= 0.5 else "NEG")
            steps += 1
        if steps >= steps_per_epoch: break
    heartbeat.write({epoch, steps, loss})

write_ndjson(tokens)

# Raincatcher metrics (cheap hash embeddings during NAS)
plv, sigma, lam = RC(tokens, prefer_st=False)

# Final objective
score = loss_final + chaos_pen(plv, sigma, lam; mode_w) - struct_bonus(spec)
return score
```

### 5.2 Raincatcher (token stream → metrics)

```python
def RC(tokens, prefer_st=True):
    embs = embed(tokens, prefer_st)   # ST model if available else hashed fallback
    s = cos_sim_series(embs)
    plv = hilbert_plv(s, window=W)
    sigma = branching_factor(tokens)
    lam = lyapunov_proxy(s)
    return plv, sigma, lam
```

---

## 6. What’s Novel

1. **Dynamics‑first NAS**: The search objective explicitly targets **emergent regimes** (PLV/σ/λ), not just error. This is a *different optimization problem* from classic NAS and yields architectures that **feel** coherent and adaptable.
2. **Receipts of loop health**: Every trial produces **auditable evidence** (JSON/MD) showing attractor behavior, criticality, and expansion/contraction. This pushes beyond single scalar benchmarks.
3. **Lightweight, pluggable measurement**: Raincatcher runs on **cheap token streams**; no intrusive tracing or heavyweight probes. Works with hashed embeddings during NAS, and higher‑fidelity embeddings post‑hoc.
4. **Structure‑dynamics coupling**: Lateral wiring modes (`affinity`, `ws`, `ba`) and allowed skips influence **σ_struct** and **modularity**, which are included in the score for **topological regularization**.
5. **Runtime compatibility**: The metrics and modes are directly reusable in deployable systems (e.g., CinderForge S) for **real‑time gating**: keep dynamics in band, enable “charisma” only when safe (λ ≤ 0), and break loops automatically.

---

## 7. Capabilities (current) & Theoretical Potential

### 7.1 Current Capabilities

- **NAS on CUDA/BF16**, seq_len 256+; support for long context via `accum_steps` + `steps_per_epoch`.
- **O(n) inference paths** through linear attention (MHLA+RoPE), suitable for production agents.
- **Modes**: observe / gentle / hard (auto planned).
- **Artifacts**: `summary.json`, `best.json`, `tokens.ndjson`, `heartbeat.ndjson`, Raincatcher JSON/MD.
- **Ops tooling**: scripts to tail heartbeat and run precise Raincatcher on newest trial.

### 7.2 Theoretical Potential (what we are testing)

- **Small‑model outsized competence**: If the search reliably steers to PLV≈0.6, σ≈1, λ≤0, even small models should exhibit **coherent multi‑step reasoning** and **clean loop exits**.
- **Generalization under perturbation**: Models selected by dynamics should **retain compositional skills** under noise or domain drift better than purely loss‑optimized peers.
- **Dial‑a‑vibe runtime**: The same metrics can drive **on‑demand “uncanny” states** (higher PLV) only when λ≤0 and σ in band—boosting perceived quality without runaway risk.

---

## 8. Early Observations (from engineering runs)

- **Gentle** mode often converged faster/lower than **hard** for `mhla_rope` on the synthetic task, suggesting strong nudging can over‑stabilize exploration.
- **Raincatcher** detected **very high PLV (~0.97)** on a sample trial with **λ < 0** and **σ ≈ 1.5**. Interpretation: a **stable deep attractor** but slightly over‑branched token transitions. For emergent‑yet‑adaptable behavior, we will aim to **lower PLV toward ~0.6** and tighten σ toward **~1.0**.
- **Heartbeat** traces confirm that `steps_per_epoch` and `accum_steps` provide effective control over runtime cadence on GPU.

*(Note: these runs used a synthetic POS/NEG tokenization; richer alphabets should sharpen the metrics.)*

---

## 9. Evaluation Plan (“win by the numbers”)

For each study, report:

1. **Convergence:** best loss, time‑to‑threshold.
2. **Stability:** % epochs with **λ ≤ 0**, % of windows with **σ ∈ [0.9, 1.1]**.
3. **Emergence band time:** % windows with **PLV ∈ [0.55, 0.70]**.
4. **Robustness@Noise:** re‑run Raincatcher with noise injection (`--noise_level`) and measure loss/PLV drift.
5. **Scale efficiency:** tokens/s and memory vs. seq_len (256→4k+), successful long‑context trials.
6. **Structure:** σ_struct near 1 with non‑trivial modularity.

The claim is *stronger* if the best architectures score well on **both** task and chaos receipts across **perturbations**.

---

## 10. Roadmap

1. **Auto mode (per‑epoch)** — Increase/decrease `rc_weight` based on rolling PLV/σ/λ. Gate “charisma” (allow higher PLV) only when λ ≤ 0 and σ in band.
2. **Chaos receipts per study** — Aggregate PLV/σ/λ distributions, % in band, and noise‑stress deltas.
3. **Richer token alphabet** — Quantize key internal stats (e.g., attention energy) into 4–8 symbols to improve sensitivity over POS/NEG.
4. **Long‑context presets** — Named settings for 4k/8k (batch, accumulation, step caps).
5. **Topology‑aware generation** — Sample lateral structures conditional on σ_struct/modularity history.
6. **Benchmark harness** — Public scripts to reproduce receipts and compare against baselines under equal budget/perturbation.

---

## 11. Safety & Operations

- **Measurement‑first**: default **observe** mode gathers dynamics before shaping.
- **Soft constraints**: nudging is incremental, avoiding hard clamping of behavior.
- **Tripwires** (deploy side): if PLV is high **and** λ>0 for consecutive windows, **break pattern** (retrieval call, reframe, or escalate).
- **Reproducibility**: studies persist trial folders with all metrics and summaries; ops scripts surface live and post‑hoc views.

---

## 12. Implementation Notes

**File structure (selected):**
```
cinderforge_e/
  raincatcher/
    __init__.py
    ingest.py
    metrics.py
    run.py                 # CLI: `raincatcher`
  search/
    cli.py                 # CLI: `cfe-study`
  trainer/
    train_seq.py           # steps_per_epoch, accum_steps, heartbeat writes
reports/
  study_<timestamp>/
    best.json
    t000/
      summary.json
      tokens.ndjson
      heartbeat.ndjson
```

**Typical commands:**

- Install after edits:  
  `pip install -e .`

- Fast sanity study (GPU, BF16):  
  `cfe-study --trials 1 --epochs 1 --device cuda --precision bf16 --seq_len 256 --batch_size 128 --steps_per_epoch 8 --accum_steps 2 --unit_type mhla_rope --rc_mode observe`

- Tail live loss (newest trial):  
  `.\Tail-LatestHeartbeat.ps1`

- Precise Raincatcher on newest trial:  
  `.\rc_latest.ps1 -PlvWindow 128 -Device cpu -Precision fp32`

---

## 13. Limitations

- **Coarse tokenization** (POS/NEG) limits granularity of PLV/σ/λ; richer alphabets are planned.
- **Embedding dependence**: SentenceTransformers introduces a distributional prior; we mitigate via hashed fallback during NAS and ST post‑hoc.
- **Search budget**: Chaos‑aware scoring adds small overhead; long‑context runs require careful budgeting (accumulation/step caps).

---

## 14. Conclusion

CinderForge E reframes NAS as **dynamics‑aware optimization**. By measuring and softly steering toward **PLV ~0.6**, **σ ~ 1**, and **λ ≤ 0**, the system selects architectures that inhabit the “**useful unusual**” regime—coherent, adaptive, and robust under perturbation. The same signals translate into deployable runtime control. The combination of **receipts**, **modes**, and **topology‑dynamics coupling** is the novelty: it elevates emergent behavior from a by‑product to a first‑class optimization target.

---

## Appendix A — Config template (excerpt)

```yaml
defaults:
  device: cuda
  precision: bf16
  seq_len: 256
  batch_size: 128
  steps_per_epoch: 16
  accum_steps: 1

raincatcher:
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  plv_window: 128
  plv_thresh: 0.7
  noise_level: 0.0

nas:
  rc_mode: observe   # observe | gentle | hard (auto planned)
  rc_weight: null
  trials: 4
  epochs: 4
  unit_type: mhla_rope
```

## Appendix B — Example Raincatcher JSON

```json
{
  "n_tokens": 64,
  "plv": 0.6421,
  "sigma": 1.03,
  "lambda_local": -0.12,
  "criticality": "balanced",
  "attractor": true,
  "plv_window": 128,
  "plv_thresh": 0.7,
  "noise_level": 0.0,
  "data_quality": "good"
}
```
