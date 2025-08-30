# cinderforge_e/launch/ray_launcher.py
from __future__ import annotations
import os, argparse, json, time
import ray
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.air import Checkpoint

def train_loop_per_worker(config):
    # Set torchrun-style env so your code path stays the same
    import os, torch, torch.distributed as dist
    from cinderforge_e.trainer.train_seq import run_training   # your existing entry

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", str(config["num_workers"])))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank % torch.cuda.device_count())))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("WORLD_SIZE", str(world))
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("LOCAL_RANK", str(local_rank))

    # Tell your trainer we are in Ray-DDP
    os.environ.setdefault("CFE_LAUNCHER", "torchrun")

    # Optional: pass nudges centrally via config (rank-safe)
    nudge = config.get("nudge", "auto")
    os.environ.setdefault("CFE_GATE_MODE", nudge)  # your code should read this into cfg.gate.mode

    # Run your normal training (reads YAML + env knobs)
    run_training()  # inside, you already do DDP init if torchrun-like env exist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--nudge", type=str, default="auto", choices=["off","observe","gentle","hard","auto"])
    ap.add_argument("--results", type=str, default="ray_results/cfe")
    args = ap.parse_args()

    ray.init(address="auto", ignore_reinit_error=True)

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={"num_workers": args.num_workers, "nudge": args.nudge},
        scaling_config=ScalingConfig(
            num_workers=args.num_workers,
            use_gpu=args.use_gpu,
            resources_per_worker={"CPU": 1, "GPU": 1},
            trainer_resources={"CPU": 1}
        ),
        run_config=RunConfig(
            name=f"cfe_{int(time.time())}",
            storage_path=args.results,
            checkpoint_config=None
        ),
    )
    result = trainer.fit()
    print("Ray Train finished:", result)

if __name__ == "__main__":
    main()
