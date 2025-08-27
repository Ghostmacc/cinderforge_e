import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

from cinderforge_e.dsl.spec import GraphSpec, LevelSpec
from cinderforge_e.gen.generate import generate_futures
from cinderforge_e.materialize.torchnet import build_model_from_spec

def load_adult():
    data = fetch_openml(name="adult", version=2, as_frame=True)
    X, y = data.data, (data.target == ">50K").astype(int)

    cat_cols = X.select_dtypes(include=["category","object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # OneHotEncoder param name changed across sklearn versions; be robust
    try:
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(X[cat_cols])
    except TypeError:
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = enc.fit_transform(X[cat_cols])

    scaler = StandardScaler()
    X_num = scaler.fit_transform(X[num_cols])

    X_all = np.hstack([X_num, X_cat]).astype(np.float32)
    y_all = y.values.astype(np.float32)
    Xtr, Xte, ytr, yte = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )
    return Xtr, Xte, ytr, yte

def train_eval(unit_type:str, lateral_mode:str, epochs:int=10, lr:float=1e-3):
    Xtr, Xte, ytr, yte = load_adult()
    d_model = Xtr.shape[1]  # input feature count

    spec = GraphSpec(
        id="adult",
        seed=7,
        levels=[LevelSpec(units=2,neurons_per=d_model),
                LevelSpec(units=3,neurons_per=d_model),
                LevelSpec(units=2,neurons_per=d_model)],
        max_vertical_skip=2, max_lateral_right=2, lateral_gate_after=1,
        resources={"unit_type":unit_type, "lateral_mode":lateral_mode, "ws_k":2, "ws_beta":0.2, "ba_m":2}
    )
    fut = generate_futures(spec)
    model = build_model_from_spec(spec, fut)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xb = torch.from_numpy(Xtr).to(dev)
    yb = torch.from_numpy(ytr).to(dev)
    model.to(dev)

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.BCELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        out = model(xb)
        loss = loss_fn(out, yb)
        loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xte).to(dev)).cpu().numpy()
        auc = roc_auc_score(yte, logits)
    return float(auc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit_type", type=str, default="mlp_block", choices=["mlp_block","linear_attn","ssm_diag"])
    ap.add_argument("--lateral_mode", type=str, default="ws", choices=["ws","ba","affinity"])
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()

    auc = train_eval(args.unit_type, args.lateral_mode, epochs=args.epochs)
    rec = {"ts": time.time(), "bench":"adult", "auc": auc,
           "unit_type": args.unit_type, "lateral_mode": args.lateral_mode, "epochs": args.epochs}
    print(json.dumps(rec, indent=2))

    os.makedirs("results", exist_ok=True)
    with open("results\\adult.ndjson","a",encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
