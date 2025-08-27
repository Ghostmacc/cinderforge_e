import uuid
import json
import time
from pathlib import Path
from typing import Optional, List

import optuna

from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.trainer.train_seq import train_one
from cinderforge_e.nas.metrics import pack_metrics

from cinderforge_e.raincatcher.ingest import read_ndjson_tokens
from cinderforge_e.raincatcher.metrics import (
    embed_tokens, compute_plv_from_embeddings, branching_factor, lyapunov_from_embeddings
)

UNIT_CHOICES = ["mlp_block", "linear_attn", "ssm_diag", "mhla_rope"]

def _spec_from_trial(trial, run_id, forced_unit_type: Optional[str]):
    d0 = trial.suggest_categorical("n0", [64, 128, 256])
    n1 = trial.suggest_categorical("n1", [64, 128, 256])
    n2 = trial.suggest_categorical("n2", [64, 128, 256])
    u0 = trial.suggest_int("u0", 2, 3)
    u1 = trial.suggest_int("u1", 3, 5)
    u2 = trial.suggest_int("u2", 2, 3)
    vskip = trial.suggest_int("vskip", 1, 2)

    if forced_unit_type and forced_unit_type != "auto":
        unit_type = forced_unit_type
    else:
        unit_type = trial.suggest_categorical("unit_type", UNIT_CHOICES)

    lateral = trial.suggest_categorical("lateral_mode", ["affinity", "ws", "ba"])
    resources = {"lateral_mode": lateral, "unit_type": unit_type, "ws_k": 2, "ws_beta": 0.2, "ba_m": 2}
    return GraphSpec(
        id=run_id, seed=int(trial.number + 1),
        levels=[
            LevelSpec(units=u0, neurons_per=d0),
            LevelSpec(units=u1, neurons_per=n1),
            LevelSpec(units=u2, neurons_per=n2),
        ],
        max_vertical_skip=vskip, max_lateral_right=2, lateral_gate_after=1, resources=resources
    )

def run_study(
    trials: int, epochs: int, out_dir: Optional[str], device: str, precision: str,
    seq_len: int, batch_size: int, unit_type: str,
    rc_mode: str, rc_weight: Optional[float],
    steps_per_epoch: Optional[int], accum_steps: int,
    rc_auto_prune: bool, rc_auto_recent: int, rc_hard_plv: float, rc_hard_sigma_dev: float
):
    mode_to_w = {"observe": 0.0, "gentle": 0.02, "hard": 0.08, "auto": 0.02}
    base_w = rc_weight if rc_weight is not None else mode_to_w.get(rc_mode, 0.0)

    study = optuna.create_study(direction="minimize")
    root_out = Path(out_dir or f"reports/study_{int(time.time())}")
    root_out.mkdir(parents=True, exist_ok=True)

    PLV_TARGET, PLV_BAND = 0.60, 0.10
    SIGMA_BAND = 0.10

    def objective(trial):
        run_id = f"trial-{uuid.uuid4().hex[:8]}"
        spec = _spec_from_trial(trial, run_id, unit_type if unit_type != "auto" else None)
        fut = generate_futures(spec)

        trial_dir = root_out / f"t{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(trial_dir / "tokens.ndjson")
        hb_path  = str(trial_dir / "heartbeat.ndjson")

        # Light-weight auto-mode adjuster (read-only; final weight chosen after training)
        def cb(epoch_idx: int, recent_tokens: List[str], last_loss: float = 0.0):
            # no heavy work here; just leave breadcrumbs if you want
            pass

        # ---- train (always log tokens + heartbeat) ----
        _, _, _, final = train_one(
            spec, fut,
            epochs=epochs, device=device, precision=precision,
            seq_len=seq_len, batch_size=batch_size,
            log_tokens=True, log_path=log_path,
            steps_per_epoch=steps_per_epoch, accum_steps=accum_steps,
            hb_path=hb_path, callback_epoch=cb, recent_window=rc_auto_recent
        )

        # Structure metrics
        struct = pack_metrics(spec, fut)
        sigma_struct = (struct.get("sigma_overall") or 1.0)
        mod = (struct.get("modularity") or 0.0)

        # Chaos metrics (token stream; hash embeddings during NAS)
        try:
            tokens = read_ndjson_tokens(log_path)
        except Exception:
            tokens = []

        if tokens:
            embs = embed_tokens(tokens, device="cpu", precision="fp32", prefer_st=False)
            plv = compute_plv_from_embeddings(embs, window=min(128, max(4, len(tokens)-1)))
            sigma_ch = branching_factor(tokens)
            lam = lyapunov_from_embeddings(embs)
        else:
            plv, sigma_ch, lam = 0.0, 1.0, 0.0

        # auto-mode: promote to hard weight for clearly sticky/divergent regimes
        w = base_w
        if rc_mode == "auto":
            if (plv >= rc_hard_plv) or (abs(sigma_ch - 1.0) > rc_hard_sigma_dev):
                w = 0.08  # hard
            else:
                w = 0.02  # gentle
            if rc_auto_prune and plv > 0.95 and sigma_ch > 1.2 and lam > 0:
                # strongly penalize degenerate thrash
                w = max(w, 0.12)

        # deviations outside dead-bands
        plv_dev = max(0.0, abs(plv - PLV_TARGET) - PLV_BAND)
        sigma_dev = max(0.0, abs(sigma_ch - 1.0) - SIGMA_BAND)
        lambda_pen = max(0.0, lam)

        chaos_pen = w * (0.5 * plv_dev + 0.5 * sigma_dev + 1.0 * lambda_pen)
        score = final + chaos_pen - 0.02 * max(0.0, sigma_struct - 1.0) - 0.01 * mod

        # persist reports
        (trial_dir / "summary.json").write_text(
            json.dumps({
                "loss": final, "score": score,
                "rc_mode": rc_mode, "rc_weight_effective": w,
                "plv": plv, "sigma_tokens": sigma_ch, "lambda_local": lam,
                "sigma_struct": sigma_struct, "modularity": mod
            }, indent=2),
            encoding="utf-8"
        )
        (trial_dir / "summary.md").write_text(
            (
                f"# Trial {trial.number}\n\n"
                f"- rc_mode: **{rc_mode}** (effective weight **{w}**)\n"
                f"- loss: **{final:.6f}**\n"
                f"- score: **{score:.6f}** (lower is better)\n\n"
                f"## Chaos metrics\n"
                f"- PLV: **{plv:.4f}** (target {PLV_TARGET}±{PLV_BAND})\n"
                f"- Sigma(tokens): **{sigma_ch:.4f}** (target 1±{SIGMA_BAND})\n"
                f"- Lambda(local): **{lam:.4f}** (penalize > 0)\n\n"
                f"## Structure metrics\n"
                f"- Sigma(struct): **{sigma_struct if sigma_struct is not None else float('nan'):.4f}**\n"
                f"- Modularity: **{mod:.4f}**\n"
            ),
            encoding="utf-8"
        )
        return score

    study.optimize(objective, n_trials=trials)
    (root_out / "best.json").write_text(
        json.dumps({"value": study.best_value, "params": study.best_trial.params}, indent=2),
        encoding="utf-8"
    )
    return str(root_out)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out_dir", type=str, default=None)
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="fp32")
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--unit_type", type=str, choices=["auto"] + UNIT_CHOICES, default="auto")

    # trainer runtime knobs
    p.add_argument("--steps_per_epoch", type=int, default=None)
    p.add_argument("--accum_steps", type=int, default=1)

    # RC knobs
    p.add_argument("--rc_mode", type=str, choices=["observe","gentle","hard","auto"], default="observe")
    p.add_argument("--rc_weight", type=float, default=None)
    p.add_argument("--rc_auto_prune", action="store_true")
    p.add_argument("--rc_auto_recent", type=int, default=64)
    p.add_argument("--rc_hard_plv", type=float, default=0.80)
    p.add_argument("--rc_hard_sigma_dev", type=float, default=0.15)

    a = p.parse_args()
    result_path = run_study(
        a.trials, a.epochs, a.out_dir, a.device, a.precision, a.seq_len, a.batch_size,
        a.unit_type, a.rc_mode, a.rc_weight, a.steps_per_epoch, a.accum_steps,
        a.rc_auto_prune, a.rc_auto_recent, a.rc_hard_plv, a.rc_hard_sigma_dev
    )
    print(f"study complete -> {result_path}")

if __name__ == "__main__":
    main()
