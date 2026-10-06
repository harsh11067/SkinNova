"""Temperature scaling fit on VAL logits by NLL (test.md C2). Writes T into the checkpoint and reports/cv_calibration.json."""
from __future__ import annotations

import json

import torch

from ml.common.paths import MODELS, REPORTS, report_meta


def ece(probs: torch.Tensor, y: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = torch.zeros(())
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (acc[m].mean() - conf[m].mean()).abs()
    return float(e)


def reliability(probs, y, bins=15):
    conf, pred = probs.max(1); acc = (pred == y).float(); out = []
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "acc": float(acc[m].mean()) if m.any() else None, "conf": float(conf[m].mean()) if m.any() else None})
    return out


def fit_temperature(logits: torch.Tensor, y: torch.Tensor) -> float:
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad(); loss = nll(logits / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp())


def main():
    import argparse
    from ml.common.paths import dataset_rev
    from ml.cv.eval_cv import load_model, logits_for
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="", help="e.g. _v2 → ckpt best_v2.pt, report cv_calibration_v2.json")
    a = ap.parse_args()
    ck_path = MODELS / "cv" / "ckpt" / f"best{a.tag}.pt"
    model, ck = load_model(ck_path)
    L, Y, _ = logits_for(model, "val", ck["classes"])
    T = fit_temperature(L, Y)
    before, after = ece(L.softmax(1), Y), ece((L / T).softmax(1), Y)
    nll = torch.nn.CrossEntropyLoss()
    ck["temperature"] = T
    torch.save(ck, ck_path)
    rep = {**report_meta(dataset_rev=dataset_rev()), "ckpt": ck_path.name, "temperature": T, "val_ece_before": before, "val_ece_after": after,
           "val_nll_before": float(nll(L, Y)), "val_nll_after": float(nll(L / T, Y)),
           "reliability_before": reliability(L.softmax(1), Y), "reliability_after": reliability((L / T).softmax(1), Y)}
    (REPORTS / f"cv_calibration{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(f"T={T:.3f} ECE {before:.4f} -> {after:.4f}")


if __name__ == "__main__":
    main()
