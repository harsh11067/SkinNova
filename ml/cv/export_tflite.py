"""Export the CV classifier to LiteRT .tflite + parity (test.md C3) + Kotlin preprocessing fixture (C4).

Runs in the isolated export venv (.venv-export: CPU torch 2.11, litert-torch 0.9.4, ai-edge-litert 2.2.0 — the
same LiteRT version as the app's com.google.ai.edge.litert:litert:2.2.0):

  .venv-export/bin/python -m ml.cv.export_tflite

Writes
  models/tflite/skin_cls.tflite                       (fp32; fp16/int8 only if their parity also passes — C3)
  android/app/src/main/assets/cv/{skin_cls.tflite, preprocess.json}, assets/model_manifest.json "cv" block
  reports/cv_parity.json                              (C3: max |Δprob| ≤ 0.01, top-1 agreement ≥ 99.5 % on 200 val)
  tests/fixtures/cv_preproc/*                         (C4: decoded RGB + exact Python input tensor for JVM tests)
  android/app/src/androidTest/assets/cv_fixtures/*    (C5: 20 images + expected probabilities for the phone)
"""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ml.common.paths import ANDROID_ASSETS, FIXTURES, MODELS, REPO, REPORTS, SPLITS, report_meta
from ml.cv.dataset import MEAN, SIZE, STD, center_square
from ml.cv.eval_cv import load_model


class NHWC(torch.nn.Module):
    """App feeds [1,384,384,3] float32 normalised RGB (preprocess.json); model wants NCHW.
    Outputs (logits [1,C], skin_logit [1,1]): the second is the skin-photo gate (ml/cv/skin_gate.py) on the same pooled
    features — p(skin photo) = sigmoid(skin_logit); below preprocess.json skin_gate.threshold the app warns."""

    def __init__(self, m, gate_w: np.ndarray, gate_b: float):
        super().__init__()
        self.m = m
        self.gate = torch.nn.Linear(gate_w.shape[0], 1)
        with torch.no_grad():
            self.gate.weight.copy_(torch.from_numpy(gate_w.astype(np.float32))[None]); self.gate.bias.fill_(float(gate_b))

    def forward(self, x):
        pooled = self.m.forward_head(self.m.forward_features(x.permute(0, 3, 1, 2)), pre_logits=True)
        return self.m.get_classifier()(pooled), self.gate(pooled)


def tfl_outputs(interp) -> tuple[int, int]:
    """(logits index, skin index) — TFLite output order is not guaranteed; tell them apart by shape."""
    outs = interp.get_output_details()
    lg = next(o for o in outs if o["shape"][-1] > 1); sk = next(o for o in outs if o["shape"][-1] == 1)
    return lg["index"], sk["index"]


def preprocess(path) -> np.ndarray:
    """Exactly dataset.eval_transform, as NHWC float32 (normalize long side ≤ 512 already holds for processed images)."""
    im = Image.open(REPO / path).convert("RGB")
    im = center_square(im).resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    return ((x - np.array(MEAN, np.float32)) / np.array(STD, np.float32))[None]


VIEWS = {"id": lambda x: x, "h": lambda x: x[:, :, ::-1, :], "v": lambda x: x[:, ::-1, :, :], "r180": lambda x: x[:, ::-1, ::-1, :]}


def tta_probs(fn, x, views, T) -> np.ndarray:
    """Mean calibrated probabilities over TTA views (NHWC ≡ torch flip(-1)/flip(-2) on NCHW; CvClassifier.view)."""
    return np.mean([fn(np.ascontiguousarray(VIEWS[v](x))) for v in views], 0)


def sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    import argparse
    import litert_torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="skin_gate.npz", help="models/cv/<file>: v2 adopted 2026-10-08 (reports/skin_gate_v2.json)")
    ap.add_argument("--tta", default="id", help="comma list of views, e.g. id,h,v,r180 (reports/cv_tta.json)")
    a = ap.parse_args(); views = a.tta.split(","); assert all(v in VIEWS for v in views)
    gate_report = "reports/" + a.gate.replace(".npz", ".json")
    from ai_edge_litert.interpreter import Interpreter
    ck_path = MODELS / "cv" / "ckpt" / "best.pt"
    model, ck = load_model(ck_path)
    assert not ck.get("color_constancy"), "CvClassifier.kt has no Shades-of-Gray yet: port it (+ C4 fixtures) before exporting this checkpoint"
    model = model.float().eval()
    T = float(ck.get("temperature", 1.0))
    g = np.load(MODELS / "cv" / a.gate); gate_t = float(g["threshold"])
    wrapped = NHWC(model, g["w"], float(g["b"])).eval()
    with torch.no_grad():   # the two-output wrapper must reproduce the classifier exactly
        x0 = torch.randn(1, SIZE, SIZE, 3); assert torch.allclose(wrapped(x0)[0], model(x0.permute(0, 3, 1, 2)), atol=1e-5)
    sample = (torch.from_numpy(preprocess(pd.read_csv(SPLITS / "val.csv").img.iloc[0])),)
    out_dir = MODELS / "tflite"; out_dir.mkdir(parents=True, exist_ok=True)
    tfl = out_dir / "skin_cls.tflite"
    litert_torch.convert(wrapped, sample).export(str(tfl))

    # ---- C3 parity on 200 val images
    va = pd.read_csv(SPLITS / "val.csv")
    va = va[va.label.isin(ck["classes"])].sample(200, random_state=3407)
    interp = Interpreter(model_path=str(tfl)); interp.allocate_tensors()
    inp = interp.get_input_details()[0]; i_lg, i_sk = tfl_outputs(interp)
    maxd, maxd_skin, agree, rows, maxd_tta = 0.0, 0.0, 0, [], 0.0
    def pt_probs(x):
        with torch.no_grad():
            return torch.softmax(wrapped(torch.from_numpy(x))[0] / T, 1).numpy()[0]
    def tfl_probs(x):
        interp.set_tensor(inp["index"], x); interp.invoke()
        return torch.softmax(torch.from_numpy(interp.get_tensor(i_lg)) / T, 1).numpy()[0]
    with torch.no_grad():
        for p in va.img:
            x = preprocess(p)
            if len(views) > 1:
                maxd_tta = max(maxd_tta, float(np.abs(tta_probs(pt_probs, x, views, T) - tta_probs(tfl_probs, x, views, T)).max()))
            lg, sk = wrapped(torch.from_numpy(x))
            pt = torch.softmax(lg / T, 1).numpy()[0]; ps = float(torch.sigmoid(sk)[0, 0])
            interp.set_tensor(inp["index"], x); interp.invoke()
            tl = torch.softmax(torch.from_numpy(interp.get_tensor(i_lg)) / T, 1).numpy()[0]
            ts = float(1 / (1 + np.exp(-interp.get_tensor(i_sk)[0, 0])))
            d = float(np.abs(pt - tl).max()); maxd = max(maxd, d); agree += int(pt.argmax() == tl.argmax())
            maxd_skin = max(maxd_skin, abs(ps - ts))
            rows.append({"img": p, "max_abs_dprob": d, "abs_dskin": abs(ps - ts)})
    tfl_sha = sha(tfl)
    from ml.common.paths import dataset_rev
    rep = {**report_meta(dataset_rev=dataset_rev(), model_sha=tfl_sha[:16]), "ckpt_sha": sha(ck_path)[:16], "tflite_sha256": tfl_sha, "tflite_bytes": tfl.stat().st_size,
           "precision": "fp32", "n": len(va), "max_abs_dprob": maxd, "top1_agreement": agree / len(va),
           "input": inp["shape"].tolist(), "input_dtype": str(inp["dtype"]), "outputs": ["logits", "skin_logit"], "max_abs_dskin_prob": maxd_skin,
           "tta_views": views, "max_abs_dprob_tta": maxd_tta, "skin_gate": a.gate}
    rep["C3_pass"] = bool(maxd <= 0.01 and rep["top1_agreement"] >= 0.995 and maxd_skin <= 0.01 and maxd_tta <= 0.01)
    (REPORTS / "cv_parity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["max_abs_dprob", "max_abs_dskin_prob", "max_abs_dprob_tta", "top1_agreement", "C3_pass", "tflite_bytes"]}))
    assert rep["C3_pass"], "C3 parity failed — do not ship this .tflite"

    # ---- ship to the app
    cv_assets = ANDROID_ASSETS / "cv"; cv_assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tfl, cv_assets / "skin_cls.tflite")
    from ml.common.paths import dataset_rev
    pre = {"size": SIZE, "mean": list(MEAN), "std": list(STD), "temperature": T, "classes": ck["classes"], "layout": "NHWC",
           "normalize_long_side": 512, "model_sha256": tfl_sha, "dataset_rev": dataset_rev(), "arch": ck["arch"],
           "skin_gate": {"output": "skin_logit", "threshold": gate_t, "report": gate_report}, "tta": views}
    (cv_assets / "preprocess.json").write_text(json.dumps(pre, indent=1))
    mf = ANDROID_ASSETS / "model_manifest.json"; m = json.loads(mf.read_text())
    m["cv"].update({"file": "cv/skin_cls.tflite", "input": [1, SIZE, SIZE, 3], "temperature": T, "sha256": tfl_sha, "classes": ck["classes"]})
    mf.write_text(json.dumps(m, indent=2) + "\n")

    # ---- C4 fixture (JVM): decoded pixels + exact Python tensor, 3 images
    fx = FIXTURES / "cv_preproc"; shutil.rmtree(fx, ignore_errors=True); fx.mkdir(parents=True)
    meta = []
    for i, p in enumerate(va.img.iloc[:3]):
        im = Image.open(REPO / p).convert("RGB")
        (fx / f"img{i}.rgb").write_bytes(np.asarray(im, np.uint8).tobytes())
        (fx / f"img{i}.f32").write_bytes(preprocess(p).astype("<f4").tobytes())
        meta.append({"name": f"img{i}", "w": im.width, "h": im.height})
    (fx / "index.json").write_text(json.dumps({"size": SIZE, "mean": list(MEAN), "std": list(STD), "images": meta}, indent=1))

    # ---- C5 fixture (phone): 20 JPEGs + expected calibrated probabilities from the .tflite
    c5 = REPO / "android/app/src/androidTest/assets/cv_fixtures"; shutil.rmtree(c5, ignore_errors=True); c5.mkdir(parents=True)
    exp = []
    for i, p in enumerate(va.img.iloc[:20]):
        shutil.copyfile(REPO / p, c5 / f"{i:02d}.jpg")   # JPEG: used as a camera-like photo by the flow tests
        # PNG = the exact pixels PIL decoded: C5 then measures model + preprocessing + TTA on the phone, not the
        # difference between Android's and PIL's JPEG decoders (that alone moved p by up to 0.04 on 2026-10-07)
        Image.open(REPO / p).convert("RGB").save(c5 / f"{i:02d}.png")
        x = preprocess(p); interp.set_tensor(inp["index"], x); interp.invoke()
        skin = float(1 / (1 + np.exp(-interp.get_tensor(i_sk)[0, 0])))   # gate: "id" view only (as the app)
        pr = tta_probs(tfl_probs, x, views, T)                            # probabilities: TTA mean (as the app)
        exp.append({"file": f"{i:02d}.png", "probs": dict(zip(ck["classes"], map(float, pr))), "skin": skin, "tta": views})
    (c5 / "expected.json").write_text(json.dumps(exp, indent=1))
    print("shipped", cv_assets / "skin_cls.tflite", tfl.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
