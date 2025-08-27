# CFE + Raincatcher — Quick Commands

## Activate & reinstall after edits
```powershell
.\.venv-gpu\Scripts\Activate.ps1
pip install -e .
```

## Fast sanity study (short, GPU if available)
```powershell
cfe-study --trials 1 --epochs 1 --device cuda --precision bf16 --seq_len 256 `
  --batch_size 128 --steps_per_epoch 8 --accum_steps 2 `
  --unit_type mhla_rope --rc_mode observe
```

## Live heartbeat in a second terminal
```powershell
.\Tail-LatestHeartbeat.ps1
```

## High-fidelity Raincatcher on newest trial
```powershell
.c_latest.ps1 -PlvWindow 128 -Device cpu -Precision fp32
```

## Read study outputs
```powershell
$latest = Get-ChildItem .eports\study_* -Directory | Sort-Object LastWriteTime -Desc | Select-Object -First 1
Get-Content (Join-Path $latest.FullName 'best.json')
Get-Content (Join-Path $latest.FullName 't000\summary.json')
```

## Long-context tip (example: 4096)
- Lower `--batch_size`
- Raise `--accum_steps`
- Cap `--steps_per_epoch`

Example:
```powershell
cfe-study --trials 2 --epochs 2 --device cuda --precision bf16 --seq_len 4096 `
  --batch_size 8 --accum_steps 4 --steps_per_epoch 24 `
  --unit_type mhla_rope --rc_mode observe
```
