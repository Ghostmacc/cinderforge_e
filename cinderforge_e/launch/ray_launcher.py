import os, argparse, ray
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer

def train_loop_per_worker(config):
    # Preserve knobs in workers.
    os.environ.setdefault("CFE_ROPE_VARIANT", config.get("rope_variant", "irope"))
    os.environ.setdefault("CFE_IROPE_GROUPS", str(config.get("irope_groups", 2)))
    os.environ.setdefault("CFE_ROPE_BASE", str(config.get("rope_base", 10000)))
    os.environ.setdefault("CFE_GATE_MODE", config.get("gate_mode", "observe"))

    # Import late so Ray workers have env set.
    from cinderforge_e.trainer.train_seq import entrypoint
    entrypoint(config.get("trainer_args", {}))

def parse_args():
    ap = argparse.ArgumentParser("CFE Ray Train Launcher")
    ap.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "8")))
    ap.add_argument("--use-gpu", action="store_true", default=True)
    ap.add_argument("--trainer-arg", action="append", default=[], help="k=v pairs")
    return ap.parse_args()

def kvs_to_dict(kvs):
    out = {}
    for kv in kvs:
        k, v = kv.split("=", 1)
        # simple coercions
        if v.isdigit(): v = int(v)
        elif v.replace('.','',1).isdigit(): v = float(v)
        out[k] = v
    return out

def main():
    a = parse_args()
    ray.init(address=os.getenv("RAY_ADDRESS", "auto"))  # works on KubeRay
    cfg = {
        "rope_variant": os.getenv("CFE_ROPE_VARIANT", "irope"),
        "irope_groups": int(os.getenv("CFE_IROPE_GROUPS", "2")),
        "rope_base": float(os.getenv("CFE_ROPE_BASE", "10000")),
        "gate_mode": os.getenv("CFE_GATE_MODE", "observe"),
        "trainer_args": kvs_to_dict(a.trainer_arg),
    }
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=cfg,
        scaling_config=ScalingConfig(num_workers=a.num_workers, use_gpu=a.use_gpu),
    )
    result = trainer.fit()
    print("Ray Train result:", result)

if __name__ == "__main__":
    main()
