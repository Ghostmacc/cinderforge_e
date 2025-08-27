import argparse
import json
from pathlib import Path
from typing import List
import torch

from cinderforge_e.raincatcher.metrics import (
    embed_tokens, compute_plv_from_embeddings, branching_factor,
    lyapunov_from_embeddings, add_noise
)
from cinderforge_e.raincatcher.ingest import read_ndjson_tokens, read_text_lines

try:
    import yaml  # optional
except Exception:
    yaml = None

def load_defaults():
    cfg_path = Path("config.yaml")
    if yaml and cfg_path.exists():
        try:
            return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}

def main():
    defaults = load_defaults()
    p = argparse.ArgumentParser(prog="raincatcher", description="PLV + σ + λ over token streams (NDJSON or text)")
    p.add_argument("--log_file", type=str, default=defaults.get("log_file"))
    p.add_argument("--text_file", type=str, default=defaults.get("text_file"))
    p.add_argument("--device", type=str, choices=["cpu","cuda"], default=defaults.get("device","cpu"))
    p.add_argument("--precision", type=str, choices=["bf16","fp16","fp32"], default=defaults.get("precision","fp32"))
    p.add_argument("--plv_window", type=int, default=int(defaults.get("plv_window",128)))
    p.add_argument("--plv_thresh", type=float, default=float(defaults.get("plv_thresh",0.7)))
    p.add_argument("--noise_level", type=float, default=float(defaults.get("noise_level",0.0)))
    p.add_argument("--st_model", type=str, default=defaults.get("embedding_model","sentence-transformers/all-MiniLM-L6-v2"))
    p.add_argument("--out_json", type=str, default=defaults.get("out_json"))
    p.add_argument("--out_md", type=str, default=defaults.get("out_md"))
    args = p.parse_args()

    if args.precision == "bf16":
        try:
            torch.set_default_dtype(torch.bfloat16)
        except Exception:
            pass

    tokens: List[str] = []
    if args.log_file:
        tokens = read_ndjson_tokens(args.log_file)
    elif args.text_file:
        tokens = read_text_lines(args.text_file)
    else:
        print("[raincatcher] ERROR: supply --log_file or --text_file")
        raise SystemExit(2)

    tokens = add_noise(tokens, args.noise_level)

    embs = embed_tokens(tokens, device=args.device, precision=args.precision, model_name=args.st_model, prefer_st=True)
    plv = compute_plv_from_embeddings(embs, window=args.plv_window)
    sigma = branching_factor(tokens)
    lam = lyapunov_from_embeddings(embs)

    summary = {
        "n_tokens": len(tokens),
        "plv": plv,
        "sigma": sigma,
        "lambda_local": lam,
        "criticality": "balanced" if 0.9 < sigma < 1.1 else "off",
        "attractor": bool(plv >= args.plv_thresh),
        "device": args.device,
        "precision": args.precision,
        "model": args.st_model,
        "plv_window": args.plv_window,
        "plv_thresh": args.plv_thresh,
        "noise_level": args.noise_level,
    }

    out = json.dumps(summary, indent=2)
    print(out)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(out, encoding="utf-8")

    if args.out_md:
        md = (
f"# Raincatcher Report\n\n"
f"- Tokens analyzed: **{len(tokens)}**\n"
f"- PLV: **{plv:.4f}** (window {args.plv_window}, thresh {args.plv_thresh})\n"
f"- Sigma (branching): **{sigma:.4f}** → {'balanced' if 0.9 < sigma < 1.1 else 'off'}\n"
f"- Lambda (local): **{lam:.4f}** → {'contractive/safe' if lam <= 0 else 'expansive/risky'}\n"
f"- Attractor detected: **{'YES' if plv >= args.plv_thresh else 'no'}**\n"
f"- Noise level: {args.noise_level}\n"
f"- Device/Precision: {args.device}/{args.precision}\n"
        )
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(md, encoding="utf-8")

if __name__ == "__main__":
    main()
