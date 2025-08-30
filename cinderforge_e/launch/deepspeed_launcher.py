import argparse, json, os, subprocess, sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser("CFE DeepSpeed Launcher")
    p.add_argument("--ds-config", default="ds_config/zero3.json")
    p.add_argument("--module", default="cinderforge_e.trainer.train_seq", help="python -m target")
    p.add_argument("--nnodes", type=int, default=int(os.getenv("WORLD_SIZE", "1")))
    p.add_argument("--nproc-per-node", type=int, default=int(os.getenv("NPROC_PER_NODE", "8")))
    p.add_argument("--node_rank", type=int, default=int(os.getenv("NODE_RANK", "0")))
    p.add_argument("--master_addr", default=os.getenv("MASTER_ADDR", "127.0.0.1"))
    p.add_argument("--master_port", default=os.getenv("MASTER_PORT", "29500"))
    p.add_argument("--train-batch-size", type=int, default=0, help="optional override")
    p.add_argument("--extra", nargs=argparse.REMAINDER, help="args passed to -m module")
    return p.parse_args()

def main():
    a = parse_args()
    cfg_path = Path(a.ds_config)
    cfg = json.loads(cfg_path.read_text())
    if a.train_batch_size > 0:
        cfg["train_batch_size"] = a.train_batch_size
        cfg_path.write_text(json.dumps(cfg, indent=2))

    # Ensure knobs survive via env (observe/gentle/hard/auto; iRoPE)
    env = os.environ.copy()
    env.setdefault("CFE_ROPE_VARIANT", env.get("CFE_ROPE_VARIANT", "irope"))
    env.setdefault("CFE_IROPE_GROUPS", env.get("CFE_IROPE_GROUPS", "2"))
    env.setdefault("CFE_ROPE_BASE", env.get("CFE_ROPE_BASE", "10000"))
    env.setdefault("CFE_GATE_MODE", env.get("CFE_GATE_MODE", "observe"))

    cmd = [
        sys.executable, "-m", "deepspeed",
        "--num_nodes", str(a.nnodes),
        "--num_gpus", str(a.nproc_per_node),
        "--master_addr", a.master_addr,
        "--master_port", a.master_port,
        "--module",
        a.module,
        "--",
    ]
    if a.extra:
        cmd += a.extra

    print("LAUNCH:", " ".join(cmd))
    sys.exit(subprocess.call(cmd, env=env))

if __name__ == "__main__":
    main()
