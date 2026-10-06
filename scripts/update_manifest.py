"""Fill the SkinNova model entry of android/app/src/main/assets/model_manifest.json from a verified .litertlm (gate G8).

  .venv/bin/python scripts/update_manifest.py models/litertlm/skinnova/skinnova-e2b-v1.litertlm \
      --l6 reports/parity_l6_skinnova_v1.json --train reports/kaggle/full/reports/full.json
Refuses unless the L6 report is for this exact file (sha256) and passed.
"""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "android/app/src/main/assets/model_manifest.json"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--l6", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--lora-source", help="where the merged adapter came from, e.g. kaggle-dataset:harsh11067/skinnova-lora-v1")
    ap.add_argument("--id", default="skinnova-e2b-v1", help="manifest entry to fill")
    ap.add_argument("--allow-l6-fail", action="store_true", help="record a model whose L6 failed (beta builds only; logged)")
    a = ap.parse_args()
    model = Path(a.model); sha = sha256(model)
    l6 = json.loads(Path(a.l6).read_text())
    assert l6["model_sha256"] == sha, "L6 report is for a different file"
    assert l6["L6"]["pass"] or a.allow_l6_fail, f"L6 did not pass: {l6['L6']}"
    tr = json.loads(Path(a.train).read_text())
    template_sha = hashlib.sha256((REPO / "ml/llm/chat_template_litertlm.jinja").read_bytes()).hexdigest()[:16]
    m = json.loads(MANIFEST.read_text())
    entry = next(e for e in m["llm"]["accepted"] if e["id"] == a.id)
    entry.update({"bytes": model.stat().st_size, "sha256": sha, "base": "google/gemma-4-E2B-it (unsloth/gemma-4-E2B-it)",
                  "lora_repo": a.lora_source or tr.get("hf_lora_repo", "kaggle:harsh11067/skinnova-gemma4-e2b-lora (output: lora/)"),
                  "converter": "litert-torch 0.9.4 export_hf + ml/llm/notebooks/export_patched.py, dynamic_wi8_emb4_afp32, vision 280",
                  "chat_template_sha": template_sha, "l6_pass": bool(l6["L6"]["pass"]),
                  "sft_dataset_rev": tr.get("dataset_meta", {}).get("git_sha", "")})
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")
    print(json.dumps(entry, indent=1))


if __name__ == "__main__":
    main()
