"""C2 metrics on val / test / external_test → reports/cv_metrics.json.

top-1, top-3, macro-F1 (bootstrap 95% CI), per-class precision/recall, confusion matrix, ECE (15 bins)
before/after temperature, per-Fitzpatrick where the source provides it. External test is scored on the
classes it actually contains (its label space is a subset).

  python -m ml.cv.eval_cv --splits val            # during development
  python -m ml.cv.eval_cv --splits val test external_test   # final candidate only (run once)
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ml.common.paths import MODELS, REPORTS, SPLITS, report_meta
from ml.cv.calibrate import ece
from ml.cv.dataset import SkinDS, eval_transform
from ml.eval.bootstrap import bootstrap_ci


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = timm.create_model(ck["arch"], pretrained=False, num_classes=len(ck["classes"]))
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m, ck


@torch.no_grad()
def logits_for(model, split, classes, bs=64, per_class: int = 0, cpu: bool = False):
    df = pd.read_csv(SPLITS / f"{split}.csv")
    df = df[df.label.isin(classes)].reset_index(drop=True)
    if per_class:
        df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), per_class), random_state=3407)).reset_index(drop=True)
    dev = "cpu" if cpu or not torch.cuda.is_available() else "cuda"
    model = model.to(dev)
    if cpu:
        torch.set_num_threads(4)
    dl = DataLoader(SkinDS(df, classes, eval_transform()), 16 if cpu else bs, num_workers=0 if cpu else 4,
                    multiprocessing_context=None if cpu else "forkserver")
    L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(dev)).float().cpu()); Y.append(y)
    return torch.cat(L), torch.cat(Y), df


def metrics(L, Y, classes, T, df):
    P = (L / T).softmax(1).numpy(); y = Y.numpy(); pred = P.argmax(1)
    present = sorted(set(y.tolist()))
    top3 = (np.argsort(-P, 1)[:, :3] == y[:, None]).any(1)
    out = {
        "n": int(len(y)),
        "top1": bootstrap_ci(lambda a, b: (a == b).mean(), pred, y),
        "top3": bootstrap_ci(lambda t: t.mean(), top3.astype(float)),
        "macro_f1_present_classes": bootstrap_ci(
            lambda a, b: f1_score(b, a, labels=present, average="macro", zero_division=0), pred, y),
        "ece_raw": ece(L.softmax(1), Y), "ece_calibrated": ece((L / T).softmax(1), Y),
        "per_class": {},
        "confusion_matrix": {"labels": classes, "matrix": confusion_matrix(y, pred, labels=list(range(len(classes)))).tolist()},
    }
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=list(range(len(classes))), zero_division=0)
    for i, k in enumerate(classes):
        if sup[i]:
            out["per_class"][k] = {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i]),
                                   "top3_recall": float(top3[y == i].mean())}
    if "skin_tone" in df and df.skin_tone.notna().any():
        out["per_skin_tone"] = {}
        for tone, g in df.groupby(df.skin_tone.fillna("unknown")):
            idx = g.index.values
            out["per_skin_tone"][str(tone)] = {"n": int(len(idx)), "top1": float((pred[idx] == y[idx]).mean()),
                                               "top3": float(top3[idx].mean())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS / "cv" / "ckpt" / "best.pt"))
    ap.add_argument("--splits", nargs="+", default=["val"])
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)
    T = ck.get("temperature", 1.0)
    sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()[:16]
    from ml.common.paths import dataset_rev
    rep = {**report_meta(dataset_rev=dataset_rev(), model_sha=sha), "arch": ck["arch"], "classes": ck["classes"], "temperature": T, "splits": {}}
    for s in a.splits:
        L, Y, df = logits_for(model, s, ck["classes"])
        rep["splits"][s] = metrics(L, Y, ck["classes"], T, df)
        m = rep["splits"][s]
        print(f"{s}: n={m['n']} top1={m['top1']['value']:.3f} top3={m['top3']['value']:.3f} "
              f"macroF1={m['macro_f1_present_classes']['value']:.3f} ECE {m['ece_raw']:.3f}->{m['ece_calibrated']:.3f}")
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "cv_metrics.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
