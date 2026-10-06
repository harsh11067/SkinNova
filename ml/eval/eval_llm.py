"""Evaluate a .litertlm on llm_val / llm_test with the real LiteRT-LM engine (same runtime version as the app: 0.17.1).

Runs in .venv-export (litert-lm-api). Same records and the same scorer (ml/eval/llm_metrics.py) as the Kaggle HF eval.
  .venv-export/bin/python -m ml.eval.eval_llm --model models/litertlm/stock/gemma-4-E2B-it.litertlm --tag stock --n 40
Writes reports/llm_litertlm_<tag>.json (+ raw outputs). Constrained JSON decoding can be compared with --constrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ml.common.paths import LLM_DATA, REPORTS, report_meta
from ml.eval.llm_metrics import ANALYSIS_KEYS, score_analysis, score_extract, summarize, summarize_extract
from ml.llm.prompt_builder import SCHEMA_COMPACT, runtime_system_message  # noqa: F401  (SCHEMA_COMPACT documents the schema)

TEMP = {"T1": 0.2, "T9": 0.2, "T6": 0.2}
MAXTOK = {"T1": 700, "T9": 700, "T6": 300}


def file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import litert_lm as L
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="llm_val"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--constrained", action="store_true"); ap.add_argument("--gray", action="store_true", help="image ablation")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(LLM_DATA / f"{a.set}.jsonl")]
    an = [r for r in recs if r["task"] in {"T1", "T9"}][:a.n]
    ex = [r for r in recs if r["task"] == "T6"][:max(5, a.n // 4)]
    t0 = time.time()
    eng = L.Engine(a.model, backend=L.Backend.CPU(), vision_backend=L.Backend.CPU(), max_num_tokens=4096)
    load_s = time.time() - t0
    gray = None
    if a.gray:
        from PIL import Image
        gray = str(Path(LLM_DATA / "gray_512.jpg")); Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)

    def run(r):
        sys_msg = r["messages"][0]["content"]
        user = r["messages"][1]["content"]
        parts = []
        for c in user:
            if c["type"] == "image":
                parts.append(L.Content.ImageFile(gray or str((LLM_DATA / r["image"]).resolve())))
            else:
                parts.append(L.Content.Text(c["text"]))
        # system message in the app's form (one-part list), so the rendered prompt is the phone's
        kw = dict(system_message=runtime_system_message(sys_msg),
                  sampler_config=L.SamplerConfig(top_k=40, top_p=0.95, temperature=TEMP[r["task"]], seed=3407),
                  thinking_config=L.ThinkingConfig(enable_thinking=False), max_output_tokens=MAXTOK[r["task"]])
        conv = eng.create_conversation(**kw)
        try:
            t = time.time()
            out = conv.send_message(L.Contents(parts))
            dt = time.time() - t
        finally:
            conv.close()
        text = "".join(c.get("text", "") for c in out.get("content", []) if isinstance(c, dict)) if isinstance(out, dict) else str(out)
        return text, dt

    rows, xrows = [], []
    for r in an:
        text, dt = run(r)
        rows.append({"id": r["id"], "task": r["task"], "s": round(dt, 2), **score_analysis(r["meta"], text), "out": text[:1500]})
        print(r["id"], rows[-1]["valid"], rows[-1]["cat_agree"], f"{dt:.1f}s", flush=True)
    for r in ex:
        text, dt = run(r)
        xrows.append({"id": r["id"], "s": round(dt, 2), **score_extract(r["meta"], text), "out": text[:600]})
    rep = {**report_meta(model_sha=file_sha(Path(a.model))[:16]), "tag": a.tag, "set": a.set, "backend": "CPU", "runtime": "litert-lm-api 0.17.1",
           "load_s": round(load_s, 1), "gray_image": bool(a.gray), "n_analysis": len(rows), "n_extract": len(xrows),
           "analysis": summarize(rows, ANALYSIS_KEYS), "extract": summarize_extract(xrows),
           "s_per_analysis_median": sorted(r["s"] for r in rows)[len(rows) // 2] if rows else None, "rows": rows, "extract_rows": xrows}
    out = REPORTS / f"llm_litertlm_{a.tag}{'_gray' if a.gray else ''}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["analysis", "extract", "load_s", "s_per_analysis_median"]}, indent=1))


if __name__ == "__main__":
    main()
