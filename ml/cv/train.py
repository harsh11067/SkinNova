"""Train the on-device classifier (plan §4: EfficientNet-B0 @384, timm, ImageNet-pretrained).

Runs locally on the RTX 3050 (6 GB) with AMP, or unchanged inside ml/cv/notebooks/cv_train.ipynb on Kaggle.
Model selection on VAL macro-F1 only. Test/external are never touched here (CLAUDE.md non-negotiable 6).

  python -m ml.cv.train                         # full run
  python -m ml.cv.train --overfit64 --epochs 30 # C1(a): must reach ≥98% train acc
  python -m ml.cv.train --random-labels --epochs 3   # C1(b): val acc must stay near chance
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from ml.common.paths import MODELS, REPORTS, SPLITS, report_meta
from ml.cv.dataset import SkinDS, eval_transform, train_transform


def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, dev):
    model.eval(); L, Y = [], []
    for x, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
            L.append(model(x.to(dev, non_blocking=True)).float().cpu())
        Y.append(y)
    return torch.cat(L), torch.cat(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--overfit64", action="store_true")
    ap.add_argument("--random-labels", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--resume", action="store_true", help="continue from models/cv/ckpt/last<tag>.pt (restart-safe)")
    a = ap.parse_args()
    set_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    classes = json.loads((SPLITS / "classes.json").read_text())["classes"]   # D5-filtered, labels.json order
    tr = pd.read_csv(SPLITS / "train.csv"); va = pd.read_csv(SPLITS / "val.csv")
    tr = tr[tr.label.isin(classes)]; va = va[va.label.isin(classes)]
    mode = "overfit64" if a.overfit64 else "random_labels" if a.random_labels else "full"
    if a.overfit64:
        tr = tr.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 6), random_state=a.seed)).head(64)
        va = tr
    tf_train = eval_transform() if a.overfit64 else train_transform()
    ds_tr = SkinDS(tr, classes, tf_train, random_labels=a.random_labels, seed=a.seed)
    ds_va = SkinDS(va, classes, eval_transform())
    counts = np.bincount(ds_tr.y, minlength=len(classes)).astype(float)
    w = 1.0 / np.maximum(counts, 1)
    sampler = None if a.overfit64 else WeightedRandomSampler([w[y] for y in ds_tr.y], num_samples=len(ds_tr), replacement=True)
    dl_tr = DataLoader(ds_tr, a.bs, sampler=sampler, shuffle=sampler is None, num_workers=a.workers,
                       pin_memory=True, drop_last=not a.overfit64, persistent_workers=True)
    dl_va = DataLoader(ds_va, a.bs * 2, num_workers=a.workers, pin_memory=True)
    model = timm.create_model(a.arch, pretrained=True, num_classes=len(classes), drop_rate=0.3).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=dev == "cuda")
    crit = nn.CrossEntropyLoss(label_smoothing=0.0 if a.overfit64 else a.smoothing)
    out_dir = MODELS / "cv" / "ckpt"; out_dir.mkdir(parents=True, exist_ok=True)
    best, hist, t0, start = -1.0, [], time.time(), 0
    last = out_dir / f"last_{mode}{a.tag}.pt"
    if a.resume and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); best, hist, start = ck["best"], ck["hist"], ck["epoch"]
        print(f"resumed from epoch {start}", flush=True)
    for ep in range(start, a.epochs):
        model.train(); tot, n, correct = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev == "cuda"):
                logits = model(x); loss = crit(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); n += len(y); correct += (logits.argmax(1) == y).sum().item()
        L, Y = predict(model, dl_va, dev)
        pred = L.argmax(1)
        acc = (pred == Y).float().mean().item()
        top3 = (L.topk(3, 1).indices == Y[:, None]).any(1).float().mean().item()
        f1 = f1_score(Y, pred, average="macro", labels=list(range(len(classes))), zero_division=0)
        present = sorted(set(Y.tolist()))
        bal = float(np.mean([(pred[Y == c] == c).float().mean().item() for c in present]))
        hist.append(dict(epoch=ep + 1, train_loss=tot / n, train_acc=correct / n, val_acc=acc, val_bal_acc=bal, val_top3=top3,
                         val_macro_f1=f1, minutes=(time.time() - t0) / 60))
        print(json.dumps(hist[-1]), flush=True)
        if mode == "full" and f1 > best:
            best = f1
            torch.save({"arch": a.arch, "classes": classes, "state_dict": model.state_dict(), "epoch": ep + 1,
                        "val_macro_f1": f1}, out_dir / f"best{a.tag}.pt")
        tmp = last.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "best": best, "hist": hist, "epoch": ep + 1}, tmp)
        tmp.replace(last)
    REPORTS.mkdir(exist_ok=True)
    from ml.data.make_splits import dataset_rev
    rep = {**report_meta(dataset_rev=dataset_rev()), "mode": mode, "args": vars(a),
           "n_train": len(ds_tr), "n_val": len(ds_va), "history": hist, "best_val_macro_f1": best,
           "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__, "timm": timm.__version__}
    if a.overfit64:
        rep["C1a_pass"] = hist[-1]["train_acc"] >= 0.98 or hist[-1]["val_acc"] >= 0.98
    if a.random_labels:
        chance = 1 / len(classes)
        rep["C1b_chance"] = chance
        # balanced accuracy, not plain accuracy: collapsing onto the largest class would look "above chance" otherwise
        rep["C1b_max_val_bal_acc"] = max(h["val_bal_acc"] for h in hist)
        rep["C1b_pass"] = rep["C1b_max_val_bal_acc"] < 2 * chance
    (REPORTS / f"cv_train_{mode}{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print("DONE", mode, {k: rep[k] for k in rep if k.startswith("C1") or k == "best_val_macro_f1"})


if __name__ == "__main__":
    main()
