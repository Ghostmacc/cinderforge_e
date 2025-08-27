# CinderForge E + Raincatcher v1.1 — Operations Guide

This guide documents the setup, knobs, and operating procedures for **CinderForge E** (NAS/search) integrated with **Raincatcher** (PLV/σ/λ metrics).

## What you get
- **Raincatcher CLI** (`raincatcher`) that reads NDJSON/token logs or text and emits JSON/MD reports:
  - PLV (phase-locking), σ (branching factor), λ (local Lyapunov proxy), and `data_quality`.
- **NAS CLI** (`cfe-study`) with chaos modes:
  - `observe` (measure-only), `gentle` (light nudging), `hard` (stronger nudging).
- **Trainer**: supports `steps_per_epoch`, `accum_steps` and writes **heartbeat.ndjson** per trial.
- **Helper scripts**:
  - `Tail-LatestHeartbeat.ps1` — tails the newest trial heartbeat.
  - `rc_latest.ps1` — precise Raincatcher on newest trial tokens.

---

## 1) One-time setup
```powershell
# Repo root
.\.venv-gpu\Scripts\Activate.ps1
pip install -e .
```

(Optional) copy config template:
```powershell
Copy-Item .\config.template.yaml .\config.yaml -Force
```

## 2) Smoke test
```powershell
# Make a tiny token file
@"
hello
hello
hello
escape
escape
novelty
"@ | Set-Content -Encoding UTF8 .\sample_tokens.txt

# Run Raincatcher
raincatcher --text_file .\sample_tokens.txt --plv_window 8 --plv_thresh 0.7 `
  --device cpu --precision fp32 `
  --out_json .esultsc_text.json --out_md .esultsc_text.md

Get-Content .esultsc_text.json
```

## 3) Fast study (sanity)
```powershell
cfe-study --trials 1 --epochs 1 --device cuda --precision bf16 --seq_len 256 `
  --batch_size 128 --steps_per_epoch 8 --accum_steps 2 `
  --unit_type mhla_rope --rc_mode observe
```

### Live monitoring in another terminal
```powershell
.\Tail-LatestHeartbeat.ps1
```

## 4) Inspect results
```powershell
$latest = Get-ChildItem .eports\study_* -Directory | Sort-Object LastWriteTime -Desc | Select-Object -First 1
Get-Content (Join-Path $latest.FullName 'best.json')
Get-Content (Join-Path $latest.FullName 't000\summary.json')

# Precise RC report for newest trial
.c_latest.ps1 -PlvWindow 128 -Device cpu -Precision fp32
```

## 5) Real runs and modes
- `observe`: dynamics are measured-only.
- `gentle`: loss + small chaos penalty (PLV/σ outside target band, λ>0).
- `hard`: stronger chaos penalty.

Example:
```powershell
cfe-study --trials 4 --epochs 4 --device cuda --precision bf16 --seq_len 256 `
  --batch_size 128 --steps_per_epoch 16 --accum_steps 1 `
  --unit_type mhla_rope --rc_mode gentle
```

## 6) Long-context guidance (e.g., 4096 tokens)
- Lower `--batch_size`
- Increase `--accum_steps`
- Cap `--steps_per_epoch` to a modest value
- Prefer `bf16` on CUDA when available

Example:
```powershell
cfe-study --trials 2 --epochs 2 --device cuda --precision bf16 --seq_len 4096 `
  --batch_size 8 --accum_steps 4 --steps_per_epoch 24 `
  --unit_type mhla_rope --rc_mode observe
```

## 7) Targets and interpretation
- **PLV** (0 to 1): 
  - Too high (≥0.80) → catatonia risk (stuck loops)
  - Too low (≤0.40) → noise
  - **Sweet spot**: ~0.55–0.70
- **σ (branching)**: aim ~1.0 (edge-of-chaos)
  - 0.9–1.1 → **balanced**
  - <0.9 rigid, >1.1 thrashing
- **λ (local Lyapunov)**:
  - λ ≤ 0 → contractive/neutral (safe)
  - λ > 0 → expansive (runaway risk)
- **data_quality** (Raincatcher):
  - low < 16 tokens < borderline < 32 tokens ≤ good

## 8) Troubleshooting
- After **any** code change affecting CLI/trainer:
  - `pip install -e .`
- `TOMLDecodeError`: replace `pyproject.toml` with minimal template, reinstall.
- `train_one() unexpected keyword`: your CLI and trainer are out of sync. Reinstall. 
- Heartbeat tail empty: start the study first, or re-run tail script to point at newest trial.

---

## Appendix: Design notes
- During NAS, we compute chaos metrics from the emitted token stream using **hash embeddings** (fast, dependency-free). Final reports use **SentenceTransformers** for accuracy.
- The chaos penalty nudges toward PLV target band and σ≈1 while penalizing λ>0. Mode weight scales the nudge.
