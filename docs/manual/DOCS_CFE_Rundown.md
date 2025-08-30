## Abbreviation Protocol (read once, follow always)

- Use common abbreviations, but **always expand the first use in each file or major section**:  
  “NAS (**Neural Architecture Search**)”, “RC (**Raincatcher**)”,  
  “PLV (**phase‑locking value**)”, “σ (**branching factor**)”, “λ (**local Lyapunov proxy**)”.
- Keep the canonical list in **docs/indices/GLOSSARY.md** and link to it when in doubt.
- After the first expansion in that file/section, short forms are fine.

# CinderForge-E: System Overview & Runbook

## 0) Mission

CinderForge-E (CFE) is our Neural Architecture Search sandbox for discovering architectures that donâ€™t just minimize loss, but also live in the right dynamics: coherent, non-brittle, and capable of â€œedge-of-chaosâ€ behavior that we can dial up or down.

Raincatcher is the dynamics meter and (optionally) a light steering wheel:

- It measures conversation/state dynamics (phase-locking, branching, Lyapunov proxy).
- In NAS mode it can nudge the search toward healthy dynamics (observe â†’ gentle â†’ hard).

The end-goal is a small, controllable model with emergent compositional behavior and runtime guardrailsâ€”and a defensible, data-driven story that outclasses Cerebros on robustness and controllability, not just raw accuracy.

## 1) Repo anatomy (whatâ€™s in `cinderforge_e/`)
```
cinderforge_e/
  dsl/              # graph/spec dataclasses (Levels, resources etc.)
  gen/              # topology generators (vertical + lateral wiring)
  nas/              # structure metrics (clustering, small-world sigma, modularity)
  units/            # neural building blocks (MLP, MHLA+RoPE, SSM-lite)
  materialize/      # turns a spec into a torch.nn.Module (ModelWrapper)
  trainer/          # train loop (toy binary sequence task); tokens logging + heartbeats
  validate/         # conformance smoke tools (determinism, structure pack)
  search/           # NAS CLI (Optuna-based), integrates Raincatcher regularizer
  raincatcher/      # PLV / branching Ïƒ / Î» metrics + CLI

tests/
pyproject.toml
requirements*.txt
README_cinderforge_raincatcher.md  # short user guide
CinderForgeE_*Whitepaper.md        # theory & competitive framing
CFE_Raincatcher_Cheatsheet.md      # quick PLV/Ïƒ/Î» reference
```

## 2) Dataflow in a study

- Spec sampling (NAS) â€“ For each Optuna trial we sample:
  - widths per level (n0,n1,n2)
  - number of units per level (u0,u1,u2)
  - vertical skip limit (vskip)
  - lateral mode (affinity, ws, ba)
  - unit type (mlp_block, linear_attn, ssm_diag, mhla_rope)

- Graph & topology â€“ `gen/generate.py` wires vertical edges (DAG) plus lateral edges per mode:
  - `ws` = small-world (Wattsâ€“Strogatz-like forward edges)
  - `ba` = scale-free (simple preferential attachment, forward only)
  - `affinity` = deterministic forward bands

- Structure sanity â€“ `nas/metrics.py` packs structure metrics:
  - avg_clustering, avg_path_len, sigma_overall (small-worldness), modularity, gini, hubs, spectral radius.

- Materialize â€“ `materialize/torchnet.py` maps the spec into a torch model:
  - Wraps a stack of chosen units around a simple head.
  - Current units:
    - `mlp_block` â€“ LN â†’ Linear â†’ GELU â†’ Dropout â†’ Linear.
    - `mhla_rope` â€“ multi-head linear attention with RoPE (fast, O(TD), causal).
    - `ssm_diag` â€“ depthwise, causal conv as a tiny SSM proxy.

- Training â€“ `trainer/train_seq.py` on a synthetic sequence classification toy:
  - Generates POS/NEG sequences; supports bf16/fp16 autocast, gradient accumulation, steps_per_epoch.
  - Logs POS/NEG tokens per optimizer step to `tokens.ndjson`.
  - Streams `heartbeat.ndjson` with {epoch,step,loss} for live tailing.

- Dynamics regularization (optional) â€“ `search/cli.py` then:
  - Reads `tokens.ndjson`.
  - Computes Raincatcher metrics quickly (hash embeddings during NAS; ST embeddings in offline reports).
  - Builds a score = loss + rc_penalty âˆ’ structure_bonus:
    - Loss (task).
    - RC penalty (observe=0, gentleâ‰ˆ0.02, hardâ‰ˆ0.08): penalizes PLV far from ~0.6, Ïƒ far from 1, Î»>0.
    - Structure bonus: small reward for small-worldness & modularity.

- Outputs â€“ `reports/study_*`:
  - `tXXX/summary.json` with loss, PLV/Ïƒ/Î», structure metrics, and final score.
  - `tXXX/tokens.ndjson` for high-fidelity Raincatcher reporting later.

## 3) Raincatcher (metrics & modes)

- File: `cinderforge_e/raincatcher/metrics.py`
- CLI: `raincatcher` (entrypoint to `raincatcher/run.py`)

### Metrics

- PLV (phase-locking value, 0â€“1):
  - From the analytic phase of the cosine-similarity time-series between successive embeddings.
  - High PLV â†’ strong attractor (coherence, but risk of loops).
  - Low PLV â†’ noise / no persistence.
  - Target band: ~0.55â€“0.70 (uncanny can allow up to 0.8 if safe).

- Ïƒ (branching factor) from token transition graph:
  - Ïƒâ‰ˆ1 â†’ edge-of-chaos; rich, adaptable computation.
  - Ïƒ<1 â†’ too rigid; Ïƒ>1.1 â†’ thrashy.

- Î» (local Lyapunov proxy) over second-difference growth of the sim time-series:
  - Î»â‰¤0 â†’ contractive/neutral (stable/escapable).
  - Î»>0 â†’ expanding (runaway risk). We only penalize positive Î».

### Modes (NAS)

- `--rc_mode observe` â€“ no regularization; score = loss âˆ’ structure_bonus.
- `--rc_mode gentle` â€“ small penalty for |PLVâˆ’0.6| beyond band Â±0.1; |Ïƒâˆ’1| beyond Â±0.1; Î»>0.
- `--rc_mode hard` â€“ stronger penalty (same targets, higher weight).

Internally, penalties are scaled by `rc_weight`; mode selects defaults but you can override.

## 4) Tuning knobs you can safely touch

NAS CLI (`cfe-study`)
```
--trials INT             # number of sampled specs
--epochs INT             # training epochs per trial
--device cpu|cuda        # cuda preferred on your 4070 Ti
--precision bf16|fp16|fp32
--seq_len INT            # sequence length (toy data)
--batch_size INT
--unit_type auto|mlp_block|linear_attn|ssm_diag|mhla_rope
--rc_mode observe|gentle|hard
--rc_weight FLOAT        # override mode; 0..1 (optional)
```

Trainer-level runtime knobs are forwarded by `search/cli.py`:
```
--steps_per_epoch INT    # optimizer steps per epoch (after accumulation)
--accum_steps INT        # gradient accumulation microbatches per step
```

Raincatcher CLI (`raincatcher`)
```
--log_file tokens.ndjson | --text_file sample.txt
--plv_window INT         # default 128
--plv_thresh FLOAT       # default 0.7 (for â€œattractorâ€ flag)
--device cpu|cuda        # only affects embedding model usage
--precision bf16|fp16|fp32
--noise_level FLOAT      # inject noise to stress test
--out_json PATH          # summary JSON
--out_md PATH            # pretty Markdown report
```

## 5) Recommended defaults (your 4070 Ti)

- Device/precision: `--device cuda --precision bf16`
- Unit type: start with `mhla_rope`
- Budget:
  - Quick smoke: `--trials 4 --epochs 4 --batch_size 128 --seq_len 256 --steps_per_epoch 8 --accum_steps 2`
  - Larger: increase trials or epochs gradually.
- RC mode: run both `observe` and `gentle` to compare; keep the one with better score and healthy PLV/Ïƒ/Î» receipts.

## 6) Typical command set

Environment (once per machine)
```
git clone https://github.com/Ghostmacc/cinderforge_e
cd cinderforge_e
# or: python -m venv .venv-gpu; .\.venv-gpu\Scripts\Activate.ps1
.\.venv-gpu\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

Run a short study
```
cfe-study `
  --trials 4 --epochs 4 `
  --device cuda --precision bf16 `
  --seq_len 256 --batch_size 128 `
  --steps_per_epoch 8 --accum_steps 2 `
  --unit_type mhla_rope `
  --rc_mode observe
```

Repeat once with `--rc_mode gentle`, compare the latest `reports/study_*/best.json` and `tXXX/summary.json`.

Live heartbeat tail
```
.\Tail-LatestHeartbeat.ps1
```

Raincatcher offline report for a trial
```
$trial = Get-ChildItem .\reports\study_* -Directory | Sort-Object LastWriteTime -Desc | Select-Object -First 1
$dir   = Get-ChildItem "$( $trial.FullName )\t*" -Directory | Sort-Object LastWriteTime -Desc | Select-Object -First 1
raincatcher --log_file (Join-Path $dir.FullName "tokens.ndjson") `
  --plv_window 64 --out_json (Join-Path $dir "rc.json") --out_md (Join-Path $dir "rc.md")
```

## 7) What makes this novel vs. Cerebros

- Dual-objective NAS: not only loss, but dynamics receipts (PLV/Ïƒ/Î»). We optimize for emergence you can control.
- Fast linear attention + RoPE block (MHLA+RoPE) for sequence modeling at O(TD).
- Search-time dynamics regularizer (gentle/hard) thatâ€™s soft and transparent.
- Runtime guardrail hooks: the same metrics can gate charisma vs stability modes in deployment.
- Receipts for every trial: NDJSON + JSON/MD summaries make brittleness obvious and reproducible.

## 8) Where we are & whatâ€™s next

Implemented
- NAS with structure metrics & sanity (no cycles, DAG).
- Units: MLP, MHLA+RoPE, SSM-lite.
- Trainer with bf16 autocast, accumulation, steps_per_epoch, tokens logging, heartbeats.
- Raincatcher metrics: PLV, Ïƒ, Î»; CLI; rc_mode observe/gentle/hard in search.
- GPU path works; 4070 Ti fully engaged.

Immediate milestones
- Longer seq presets (1kâ€“4k) + tune accumulation.
- Auto mode (optional): switch to hard when PLV spikes and Î»>0 for a few windows.
- More unit diversity (e.g., gated MLP, conv-mixer, recurrent SSM).
- Dataset adapters for real CSR flows (replace toy data).
- Report packer: auto-export â€œbest trial + receiptsâ€ into a single folder for sharing.

## 9) Interpreting your results (quick heuristics)

- Loss â†“ is necessary but not sufficientâ€”always check:
  - PLV in 0.55â€“0.70 (ok), up to 0.8 if Î»â‰¤0 (uncanny safe).
  - Ïƒ(tokens) around 1.0 Â± 0.1.
  - Î» â‰¤ 0 (expansive Î»>0 is penalized).
- If PLV very high and Î»>0, youâ€™re in a magnetic loopâ€”break pattern or lighten the regularizer.
- If Ïƒ >> 1, lower temperature / reduce branching (or strengthen penalty).
- Prefer trials whose structure also looks good (reasonable sigma_overall, non-trivial modularity).

## 10) Minimal mental model

Think of CFE as a spec â†’ graph â†’ model â†’ dynamics meter loop:

```
spec ~trial params~ â†’ topology â†’ model (units) â†’ train (bf16) â†’ tokens.ndjson
                                               â†˜ structure metrics

tokens.ndjson â†’ Raincatcher â†’ PLV/Ïƒ/Î» â†’ score penalty (gentle|hard) â†’ NAS feedback
```

Weâ€™re optimizing for emergence (coherent but escapable attractors, critical branching) and for controllability (knobs that can later gate runtime behavior).

## If you get stuck

- Env: make sure `(.venv-gpu)` shows in your prompt and `cfe-study` / `raincatcher` resolve:
  - `Get-Command cfe-study`
  - `Get-Command raincatcher`
- Heartbeat shows nothing? It appears only while a trial is actually training; re-run `Tail-LatestHeartbeat.ps1` right after you launch a study.
- Large zips / git bloat: venvs, caches, `reports/`, `results/` are ignored by our `.gitignore`. Use the small â€œcode-onlyâ€ zip strategy when you need a snapshot.

---

Youâ€™re not losing anything. The repo now has clean code, repeatable CLIs, and receipts; the â€œcode-onlyâ€ zips you made are a safe offline fallback. From here the fastest progress is: (1) short observe/gentle studies, (2) compare receipts, (3) lock in MHLA+RoPE defaults, (4) push a longer-seq run with accumulation, (5) export the â€œbest trial kitâ€ (model + rc.md + summary.json).

