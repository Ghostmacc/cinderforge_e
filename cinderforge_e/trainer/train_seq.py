import json
import time
from pathlib import Path
from typing import Optional, Callable, List
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from cinderforge_e.materialize.torchnet import build_model_from_spec


def _make_data_seq(dim=16, n_per=64, seq_len=128, seed=None):
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(int(seed))
    a = torch.randn(n_per, seq_len, dim, generator=g) + 1.0  # class 1
    b = torch.randn(n_per, seq_len, dim, generator=g) - 1.0  # class 0
    x = torch.cat([a, b], dim=0)
    y = torch.cat([torch.ones(n_per), torch.zeros(n_per)], dim=0)
    idx = torch.randperm(x.size(0), generator=g)
    return x[idx], y[idx]


def _write_ndjson(tokens, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for t in tokens:
            f.write(json.dumps({"token_output": str(t)}) + "\n")


def _append_heartbeat(hb_path: Optional[str], epoch: int, step: int, loss_val: float):
    if not hb_path:
        return
    p = Path(hb_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), "epoch": epoch, "step": step, "loss": loss_val}) + "\n")


def _calc_n_per(batch_size: int, steps_per_epoch: Optional[int], accum_steps: int) -> int:
    """
    Ensure at least (steps_per_epoch * accum_steps) mini-batches per epoch.
    Dataset has 2*n_per examples total; mini-batches = (2*n_per)/batch_size.
    """
    if steps_per_epoch is None:
        return max(64, batch_size)
    need_micro = batch_size * steps_per_epoch * max(1, accum_steps)
    n_per = max(batch_size, need_micro // 2)  # per class
    return int(n_per)


def train_one(
    spec,
    fut,
    epochs: int = 6,
    device: str = "cpu",
    precision: str = "fp32",
    seq_len: int = 128,
    batch_size: int = 64,
    # Raincatcher logging
    log_tokens: bool = False,
    log_path: Optional[str] = None,
    # runtime & live progress
    steps_per_epoch: Optional[int] = None,
    accum_steps: int = 1,
    hb_path: Optional[str] = None,
    # auto-mode callback (optional)
    callback_epoch: Optional[Callable[[int, List[str], float], None]] = None,
    recent_window: int = 64,
):
    dev = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, None)

    model = build_model_from_spec(spec, fut).to(dev)

    # size dataset to satisfy steps_per_epoch * accum_steps
    n_per = _calc_n_per(batch_size, steps_per_epoch, accum_steps)
    xb, yb = _make_data_seq(dim=spec.levels[0].neurons_per, n_per=n_per, seq_len=seq_len, seed=spec.seed)
    ds = TensorDataset(xb, yb.float())
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    loss_fn = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda") if (dev.type == "cuda" and amp_dtype is torch.float16) else None

    model.train()
    token_stream: List[str] = []
    recent = deque(maxlen=max(4, int(recent_window)))
    last_loss = 0.0
    global_micro = 0  # counts mini-batches

    for ep in range(epochs):
        step_in_epoch = 0
        opt.zero_grad(set_to_none=True)

        # Loop until we've performed the requested number of optimizer steps in this epoch,
        # or once through the dataset if steps_per_epoch is None.
        loader_iter = iter(dl)
        while True:
            try:
                xb_i, yb_i = next(loader_iter)
            except StopIteration:
                if steps_per_epoch is None:
                    break  # one pass over data
                loader_iter = iter(dl)
                xb_i, yb_i = next(loader_iter)

            xb_i = xb_i.to(dev, non_blocking=True)
            yb_i = yb_i.to(dev, non_blocking=True)

            if dev.type == "cuda" and amp_dtype is not None:
                with torch.amp.autocast(dev.type, dtype=amp_dtype):
                    logits = model(xb_i)
                    loss = loss_fn(logits, yb_i)
            else:
                logits = model(xb_i)
                loss = loss_fn(logits, yb_i)

            # gradient accumulation
            loss_scaled = loss / max(1, accum_steps)
            if scaler is not None:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            global_micro += 1
            # take an optimizer step every accum_steps mini-batches
            if (global_micro % max(1, accum_steps)) == 0:
                if scaler is not None:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

                with torch.no_grad():
                    p = torch.sigmoid(logits).mean().item()
                    tok = "POS" if p >= 0.5 else "NEG"
                    token_stream.append(tok)
                    recent.append(tok)
                    last_loss = float(loss.item())

                _append_heartbeat(hb_path, ep, step_in_epoch, float(last_loss))
                step_in_epoch += 1

                if steps_per_epoch is not None and step_in_epoch >= steps_per_epoch:
                    break  # this epoch reached the requested number of optimizer steps

        # end-of-epoch callback
        if callback_epoch is not None:
            try:
                callback_epoch(ep, list(recent), last_loss)
            except Exception:
                pass

    if log_tokens and log_path:
        _write_ndjson(token_stream, log_path)

    return model, None, None, last_loss
