# CinderForge E + Raincatcher — Full Session Report (2025-08-26)

## Objectives (recap)
1. **Measure** emergent dynamics (not just loss) using PLV/σ/λ.
2. **Enable controllable regimes** via NAS scoring modes:
   - `observe` (measurement only), `gentle`, `hard`.
3. **Operationalize**: provide a CLI + scripts so studies and reports are easy to run & monitor.
4. **Scale** toward long context (seq_len≫1k) without OOM.
5. **Prepare** for runtime policy (CinderForge S): gating by PLV/σ/λ bands for uncanny-but-safe behavior.

## What we implemented
- **Raincatcher package** (`cinderforge_e/raincatcher`)
  - `metrics.py`: hash/ST embeddings; PLV, σ, λ; noise; data_quality.
  - `run.py`: CLI `raincatcher` → JSON + Markdown reports.
  - `ingest.py`: NDJSON + text readers.
- **NAS integration (`cfe-study`)**
  - New args: `--rc_mode observe|gentle|hard`, optional `--rc_weight` override.
  - Trial artifacts: `tokens.ndjson`, `summary.json`, `best.json`.
  - Scoring: loss + chaos penalty (PLV dev from target, σ dev from 1, and λ>0). 
- **Trainer**
  - Knobs: `steps_per_epoch`, `accum_steps`.
  - Emits `heartbeat.ndjson` per trial; token stream (POS/NEG) for Raincatcher.
- **Ops scripts**
  - `Tail-LatestHeartbeat.ps1`
  - `rc_latest.ps1`
  - README + cheatsheet

## Evidence from the session
- **Raincatcher** on toy text:
  - PLV ~0.922 — strong attractor lock, σ=1.25, λ positive earlier; later runs showed λ negative.
- **Study runs (bf16, cuda, seq_len=256, mhla_rope)**
  - Observe/gentle/hard modes executed. Examples observed:
    - Gentle best loss ~0.1486 (earlier run), Hard best loss ~0.4797, Observe runs ~0.13–0.74 depending on spec.
  - Example hard trial: `plv=0.6607`, `sigma_tokens=1.5` (outside balanced band), `lambda_local=-5.51` (contractive), `sigma_struct=0.806` (sub-critical structure), `modularity~0.04` (low communities).
- **rc_precise** on trial tokens with ST embeddings produced PLV ~0.968 (very strong attractor).

**Interpretation:**
- When PLV is very high (≥0.90) and λ≤0, we see a **deep attractor** that is contractive (stable) but may reduce exploration (σ elevated suggests branching in tokens; structural σ below 1 indicates constrained topology). For exploration-heavy tasks, favor **PLV ~0.55–0.70** and **σ ~ 1**.
- Gentle nudging generally improved convergence compared to hard nudging for `mhla_rope` under current settings.

## Immediate milestones (high impact)
1. **Auto mode (rc_auto)** — measurement-first switching:
   - Policy: If PLV>0.80 and λ≤0 for K consecutive windows → temporarily increase regularizer weight; if λ>0 or σ deviates strongly → reduce weight or force retrieval/branching in the data stream.
   - Implementation: add `--rc_mode auto` in `cfe-study`; no trainer callback needed—use per-epoch token stream to adjust the *next* epoch’s chaos weight.
2. **Long-context preset** (4096+ tokens)
   - Provide a preset launcher with: `bf16`, `batch_size<=8`, `accum_steps>=4`, `steps_per_epoch in [16,32]`.
   - Optional hash-embeddings during NAS; precise embeddings only in post-hoc RC runs.
3. **Structured chaos receipts**
   - Aggregate over trials: write `study_*/chaos_summary.json` with stats (median PLV/σ/λ, % within band).
   - Quick dashboard script to rank trials by a combined **Stability@Noise** score.
4. **σ structural tuning**
   - Sample lateral wiring (affinity/ws/ba) conditional on σ_struct history to push toward ~1.0 while retaining modularity.
5. **Runtime S hooks (preview)**
   - Simple rule set: allow PLV to climb only if λ≤0 and σ in [0.9,1.1]; otherwise trigger tool/retrieval or temperature drop.

## Hypotheses to get closer to the end goal
- **H1—Band targeting improves robustness**: Trials that maintain PLV in 0.55–0.70 and σ≈1 across noise settings will show smaller performance degradation when perturbed.
- **H2—Gentle > Hard for mhla_rope**: Under current synthetic task and resource settings, gentle nudging converges faster and reaches lower loss; hard nudging may be too restrictive and reduce exploration.
- **H3—Structural σ matters**: Combining token σ (behavioral) with σ_struct (topological) yields better selection than either alone—look for **σ_struct ≈ 1** and **moderate modularity**.
- **H4—Auto mode reduces manual retuning**: Per-epoch weight adjustment informed by PLV/λ trends will reduce the number of trials needed to hit the sweet spot band.

## Gaps / To do
- Add `--rc_mode auto` to CLI + objective (no callbacks).
- Provide `study_*/chaos_summary.json` and a `rank_trials.ps1` helper.
- Expand trainer tokenization beyond POS/NEG to capture richer micro-dynamics (e.g., 4–8 symbol alphabet based on logit bands).
- Optional: unify config reading in `cfe-study` (we already use config for `raincatcher`).

## Practical target bands (rev)
- PLV: **0.55–0.70** (uncanny up to **0.80** only if **λ≤0**).
- σ tokens: **0.95–1.05**.
- λ: **≤0** (penalize positive only).
- σ_struct: nudge toward **~1.0** while watching modularity.

## Runbook summary
1. Activate + reinstall after edits: `pip install -e .`
2. Start a short study in `observe` (sanity), tail heartbeat with `Tail-LatestHeartbeat.ps1`.
3. Generate RC reports with `rc_latest.ps1`.
4. Try `gentle` and `hard`, compare **loss + chaos receipts**.
5. For long context, apply the long-context preset.
6. Iterate weights or move to **auto** once added.

---

This report synthesizes the full session and cements a repeatable workflow. You now have the tooling to measure, nudge, and scale dynamics—so studies don’t just chase loss, they **optimize for emergence** in a controlled, auditable way.
