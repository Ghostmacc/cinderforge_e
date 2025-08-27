import argparse, json, os, time
import torch, torch.nn as nn, torch.optim as optim

from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.materialize.torchnet import build_model_from_spec

def make_parity(N=4096, D=256, seed=0):
    g = torch.Generator().manual_seed(int(seed))
    x = torch.randint(0,2,(N,D),generator=g).float()
    y = (x.sum(dim=1) % 2).float()
    return x, y

def train_eval(unit_type:str, d_model:int=256, epochs:int=8, lr:float=1e-3):
    spec = GraphSpec(
        id="parity", seed=7,
        levels=[LevelSpec(units=3, neurons_per=d_model)],
        max_vertical_skip=1, max_lateral_right=2, lateral_gate_after=1,
        resources={"unit_type":unit_type, "lateral_mode":"ws", "ws_k":2, "ws_beta":0.2, "ba_m":2}
    )
    fut = generate_futures(spec)
    model = build_model_from_spec(spec, fut)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xb,yb = make_parity(N=4096, D=d_model, seed=spec.seed)
    xb,yb = xb.to(dev), yb.to(dev)
    model.to(dev)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(xb); loss = loss_fn(out,yb)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        preds = (model(xb) > 0.5).float()
        acc = (preds == yb).float().mean().item()
    return float(acc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit_type", type=str, default="linear_attn", choices=["mlp_block","linear_attn","ssm_diag"])
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    acc = train_eval(args.unit_type, d_model=args.d_model, epochs=args.epochs)
    rec = {"ts": time.time(), "bench":"parity", "acc": acc,
           "unit_type": args.unit_type, "d_model": args.d_model, "epochs": args.epochs}
    print(json.dumps(rec, indent=2))

    os.makedirs("results", exist_ok=True)
    with open("results\\parity.ndjson","a",encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
