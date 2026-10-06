"""L6 export parity (test.md §5) on the PC: HF merged model (E1 greedy outputs from the Kaggle training report) vs a
.litertlm (greedy, LiteRT-LM 0.17.1), same llm_val prompts, plus the gray-image ablation on the first 16.

  .venv-export/bin/python -m ml.eval.parity_l6 --model models/litertlm/hybrid/skinnova-hybrid.litertlm \
      --hf-report reports/kaggle/full/reports/full.json --tag hybrid_v1 --threads 12
Pass: JSON validity drop ≤ 2 pts, category-agreement drop ≤ 3 pts, tier compliance 100 %, image dependence ≥ 75 % of the
HF rate, every output starts with "{". Writes reports/parity_l6_<tag>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from ml.common.paths import LLM_DATA, REPORTS, report_meta
from ml.eval.llm_metrics import ANALYSIS_KEYS, score_analysis, summarize
from ml.llm.prompt_builder import runtime_system_message


def main():
    import litert_lm as L
    from PIL import Image
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--hf-report", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--n-gray", type=int, default=16)
    a = ap.parse_args()
    hf = json.load(open(a.hf_report))
    e1 = hf["L4_analysis_rows"]["E1"]
    # the val records this LoRA was evaluated on: the SFT build named in the training report. Ids are reused across
    # rebuilds, so match the build (created_at), not the ids: current data/llm, else the frozen v1 copy in data/llm_eval.
    built = hf.get("dataset_meta", {}).get("created_at")
    current = json.loads((REPORTS / "sft_data.json").read_text()).get("created_at")
    frozen = LLM_DATA.parent / "llm_eval"
    base = LLM_DATA if built == current else frozen
    if base == frozen:
        assert json.loads((frozen / "sft_data_v1.json").read_text()).get("created_at") == built, "no copy of this LoRA's val set"
    recs = {json.loads(l)["id"]: json.loads(l) for l in open(base / "llm_val.jsonl")}
    missing = [h["id"] for h in e1 if h["id"] not in recs]
    assert not missing, f"E1 records not found in {base}: {missing[:5]}"
    cpu = (lambda: L.Backend.CPU(thread_count=a.threads)) if a.threads else (lambda: L.Backend.CPU())
    caps = L.Capabilities(a.model); m = caps.input_modalities; has_vision, has_audio = bool(m.vision), bool(m.audio); caps.close()
    t0 = time.time()
    eng = L.Engine(a.model, backend=cpu(), vision_backend=cpu() if has_vision else None, max_num_tokens=4096)
    load_s = time.time() - t0
    gray = str(LLM_DATA / "gray_512.jpg"); Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)

    def ask(r, image=None):
        parts = [L.Content.ImageFile(image or str((base / r["image"]).resolve())) if c["type"] == "image" else L.Content.Text(c["text"])
                 for c in r["messages"][1]["content"]]
        conv = eng.create_conversation(system_message=runtime_system_message(r["messages"][0]["content"]),
                                       sampler_config=L.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=3407),
                                       thinking_config=L.ThinkingConfig(enable_thinking=False), max_output_tokens=700)
        try:
            t = time.time(); out = conv.send_message(L.Contents(parts)); dt = time.time() - t
        finally:
            conv.close()
        return "".join(c.get("text", "") for c in out.get("content", []) if isinstance(c, dict)), dt

    rows = []
    for i, h in enumerate(e1):
        r = recs[h["id"]]
        txt, dt = ask(r)
        row = {"id": r["id"], "s": round(dt, 1), **score_analysis(r["meta"], txt), "out": txt, "hf_cats": h["cats"],
               "same_cats_as_hf": set(score_analysis(r["meta"], txt)["cats"]) == set(h["cats"])}
        if i < a.n_gray and has_vision:
            g, _ = ask(r, gray); row["gray_changed"] = set(score_analysis(r["meta"], g)["cats"]) != set(row["cats"])
        rows.append(row)
        print(r["id"], "valid", row["valid"], "agree", row["cat_agree"], "same_as_hf", row["same_cats_as_hf"], f"{dt:.0f}s", flush=True)
    hf_s, lt_s = summarize(e1, ANALYSIS_KEYS), summarize(rows, ANALYSIS_KEYS)
    g = [x["gray_changed"] for x in rows if "gray_changed" in x]
    hf_abl = hf.get("image_ablation", {}).get("changed_rate")
    L6 = {"n": len(rows), "hf": hf_s, "litertlm": lt_s,
          "json_validity_drop": round(hf_s["valid"] - lt_s["valid"], 4), "category_agreement_drop": round(hf_s["cat_agree"] - lt_s["cat_agree"], 4),
          "category_sets_identical_to_hf": round(sum(x["same_cats_as_hf"] for x in rows) / len(rows), 4), "tier_ok": lt_s["tier_ok"],
          "image_ablation_hf": hf_abl, "image_ablation_litertlm": round(sum(g) / len(g), 4) if g else None,
          "starts_with_brace": round(sum(x["out"].lstrip().startswith("{") for x in rows) / len(rows), 4),
          "capabilities": {"vision": has_vision, "audio": has_audio}, "load_s": round(load_s, 1),
          "s_per_case_median": sorted(x["s"] for x in rows)[len(rows) // 2]}
    L6["pass"] = bool(L6["json_validity_drop"] <= 0.02 and L6["category_agreement_drop"] <= 0.03 and L6["tier_ok"] == 1.0
                      and (not hf_abl or (L6["image_ablation_litertlm"] or 0) >= 0.75 * hf_abl) and L6["starts_with_brace"] == 1.0)
    h = hashlib.sha256()
    with open(a.model, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    rep = {**report_meta(model_sha=h.hexdigest()[:16]), "tag": a.tag, "model": str(a.model), "model_sha256": h.hexdigest(),
           "hf_report": a.hf_report, "L6": L6, "rows": rows}
    (REPORTS / f"parity_l6_{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(L6, indent=1))


if __name__ == "__main__":
    main()
