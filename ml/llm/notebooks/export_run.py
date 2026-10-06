"""Runs INSIDE the Kaggle export notebook's isolated venv (see make_export.py): export → capabilities → L6-style suite.
Usage there: /tmp/ve/bin/python /tmp/run.py <stock|merged>"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

MODE = sys.argv[1]
DATA = os.path.dirname(glob.glob("/kaggle/input/**/sft_train.jsonl", recursive=True)[0])
sys.path.insert(0, DATA + "/code")
from ml.eval.llm_metrics import ANALYSIS_KEYS, score_analysis, summarize  # noqa: E402

OUT = "/kaggle/working"
os.makedirs(f"{OUT}/reports", exist_ok=True); os.makedirs(f"{OUT}/export", exist_ok=True)
REPORT = {"mode": MODE, "steps": []}


def save():
    json.dump(REPORT, open(f"{OUT}/reports/export_{MODE}.json", "w"), indent=1, default=str)


from huggingface_hub import hf_hub_download, snapshot_download  # noqa: E402

MERGE_PY = r"""
# Streaming manual LoRA merge: W <- W + (alpha/r) * B @ A for every adapted layer (fp32 maths, base dtype out), one tensor
# at a time, written as ~2 GB shards + model.safetensors.index.json — peak RAM ~2 GB. (Export v13 loaded the whole 10 GB
# checkpoint plus its serialised copy and the container was killed without a log. PEFT cannot wrap Gemma4ClippableLinear.)
import glob, json, os, shutil, sys, time, torch
from safetensors import safe_open
from safetensors.torch import save_file
base, adapter, out = sys.argv[1:4]
log = open(os.environ.get("MERGE_LOG", "/kaggle/working/merge.log"), "a")
def say(*a):
    print(*a, flush=True); print(time.strftime("%H:%M:%S"), *a, file=log, flush=True)
cfg = json.load(open(os.path.join(adapter, "adapter_config.json")))
scale = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
A, B = {}, {}
with safe_open(os.path.join(adapter, "adapter_model.safetensors"), "pt") as f:
    for k in f.keys():
        mod = k.replace("base_model.model.", "", 1).split(".lora_")[0]
        (A if ".lora_A." in k else B)[mod] = f.get_tensor(k).float()
assert set(A) == set(B), "unpaired LoRA tensors"
os.makedirs(out, exist_ok=True)
done, wmap, buf, size, part, total = set(), {}, {}, 0, 0, 0
def flush():
    global buf, size, part
    if not buf:
        return
    name = f"model-{part:05d}.safetensors"; save_file(buf, os.path.join(out, name), metadata={"format": "pt"})
    for k in buf: wmap[k] = name
    say(f"wrote {name}: {len(buf)} tensors, {size / 1e9:.2f} GB"); buf, size, part = {}, 0, part + 1
for shard in sorted(glob.glob(os.path.join(base, "*.safetensors"))):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            t = f.get_tensor(k); mod = k[:-len(".weight")] if k.endswith(".weight") else None
            if mod in A:
                t = (t.float() + scale * (B[mod] @ A[mod])).to(t.dtype); done.add(mod)
            buf[k] = t.contiguous(); n = t.numel() * t.element_size(); size += n; total += n
            if size >= 2e9:
                flush()
flush()
missing = set(A) - done
assert not missing, f"{len(missing)} adapted modules not found in the base checkpoint, e.g. {sorted(missing)[:3]}"
json.dump({"metadata": {"total_size": total}, "weight_map": wmap}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
for f in glob.glob(os.path.join(base, "*.json")) + glob.glob(os.path.join(base, "*.jinja")):
    if not os.path.basename(f).startswith("model.safetensors"):
        shutil.copy(f, out)
say(f"merged {len(done)} weights (scale {scale}) -> {out}, {total / 1e9:.2f} GB in {part} shards")
"""

if MODE == "stock":
    SRC = snapshot_download("unsloth/gemma-4-E2B-it", local_dir="/tmp/gemma4_e2b")
elif MODE.startswith("lora"):   # adapter dataset (adapter_config.json + reports/full.json) → merge on this 32 GB CPU machine
    base = snapshot_download("unsloth/gemma-4-E2B-it", local_dir="/tmp/gemma4_e2b")
    adapter = os.path.dirname(glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)[0])
    open("/tmp/merge.py", "w").write(MERGE_PY)
    t = time.time(); mp = subprocess.run([sys.executable, "/tmp/merge.py", base, adapter, "/tmp/merged"], capture_output=True, text=True)
    REPORT["merge"] = {"rc": mp.returncode, "s": round(time.time() - t), "adapter": adapter, "tail": (mp.stdout + mp.stderr)[-1500:]}; save()
    assert mp.returncode == 0, REPORT["merge"]["tail"]
    SRC = "/tmp/merged"
else:
    SRC = os.path.dirname(glob.glob("/kaggle/input/**/merged/config.json", recursive=True)[0])
REPORT["source"] = SRC; save(); print("source", SRC, flush=True)


import threading  # noqa: E402

MEM = []


def _memlog():
    while True:
        try:
            info = dict(l.split(":", 1) for l in open("/proc/meminfo").read().splitlines())
            MEM.append((round(time.time()), int(info["MemTotal"].split()[0]) // 1024, int(info["MemAvailable"].split()[0]) // 1024))
        except Exception:  # noqa: BLE001
            pass
        time.sleep(2)


threading.Thread(target=_memlog, daemon=True).start()


PATCHED = os.path.join(os.path.dirname(os.path.abspath(__file__)), "export_patched.py")   # memory fixes, see that file


def export():
    out = "/tmp/exp"
    subprocess.run(["rm", "-rf", out])
    # --export_audio_encoder is read only by the Qwen3 ASR path in litert-torch 0.9.4; Gemma 4 gets no audio either way.
    cmd = [sys.executable, PATCHED, "export_hf", SRC, out, "--task=image_text_to_text",
           "--externalize_embedder=True", "--quantization_recipe=dynamic_wi8_emb4_afp32", "--export_vision_encoder=True",
           "--export_audio_encoder=False", "--bundle_litert_lm=True", "--use_jinja_template=True",
           "--experimental_lightweight_conversion=True",
           # LiteRT-LM's Jinja engine rejects the HF repo template ("unknown method", export v9) → bundle the template from
           # Google's official .litertlm, which is also the template the LoRA was trained with
           f"--jinja_chat_template_override={DATA}/code/chat_template_litertlm.jinja",
           # vision budget = training's (HF Gemma4ImageProcessor: image_seq_length 280 → ≤ 2,520 patches); litert-torch 0.9.4
           # defaults to 140 soft tokens (max_num_patches 1,260), i.e. half the image detail the model was trained on
           "--gemma4_vision_max_soft_tokens=280"]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    files = glob.glob(f"{out}/**/*.litertlm", recursive=True)
    REPORT["steps"].append({"rc": p.returncode, "s": round(time.time() - t), "files": files,
                            "mem_total_mb": MEM[-1][1] if MEM else None, "mem_min_available_mb": min((m[2] for m in MEM), default=None),
                            "released": [l for l in p.stdout.splitlines() if l.startswith("[skinnova]")],
                            "stdout_tail": p.stdout[-1500:], "stderr_tail": p.stderr[-3000:]}); save()
    print(f"export rc={p.returncode} files={files} ({round(time.time() - t)}s)", flush=True)
    if p.returncode:
        print(p.stderr[-2500:], flush=True)
    return files[0] if files else None


model = export()
if not model:
    raise SystemExit(f"export failed; see reports/export_{MODE}.json")
dst = f"{OUT}/export/skinnova-e2b-{MODE}.litertlm"
subprocess.run(["cp", model, dst])
h = hashlib.sha256()
with open(dst, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
REPORT["litertlm"] = {"path": dst, "bytes": os.path.getsize(dst), "sha256": h.hexdigest()}; save(); print(REPORT["litertlm"], flush=True)

import litert_lm as L  # noqa: E402
from PIL import Image  # noqa: E402


def caps(path):
    c = L.Capabilities(path); m = c.input_modalities   # a property in litert-lm-api 0.17.1 (Kotlin: inputModalities())
    r = {"text": m.text, "vision": m.vision, "audio": m.audio}; c.close()
    return r


def load_engine(path):
    cap = caps(path)
    t = time.time()
    eng = L.Engine(path, backend=L.Backend.CPU(), vision_backend=L.Backend.CPU() if cap["vision"] else None,
                   audio_backend=L.Backend.CPU() if cap["audio"] else None, max_num_tokens=4096)
    return eng, cap, round(time.time() - t, 1)


def ask(eng, parts, system=None, max_tok=700):
    # greedy (parity with the HF greedy outputs); system message in the app's form (one-part list, C API parses JSON)
    sm = json.dumps([{"type": "text", "text": system}], ensure_ascii=False) if system else None
    conv = eng.create_conversation(system_message=sm, sampler_config=L.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=3407),
                                   thinking_config=L.ThinkingConfig(enable_thinking=False), max_output_tokens=max_tok)
    try:
        out = conv.send_message(L.Contents(parts))
    finally:
        conv.close()
    return "".join(c.get("text", "") for c in out.get("content", []) if isinstance(c, dict))


def run_suite(path, tag, cases, n_gray):
    eng, cap, load_s = load_engine(path)
    res = {"load_s": load_s, "capabilities": cap, "text_probe": ask(eng, [L.Content.Text("Reply with exactly the word OK.")], max_tok=8)}
    gray = "/tmp/gray.jpg"; Image.new("RGB", (512, 512), (128, 128, 128)).save(gray)
    rows = []
    for i, r in enumerate(cases):
        parts, kinds = [], []
        for c in r["messages"][1]["content"]:
            parts.append(L.Content.ImageFile(f'{VAL_DIR}/{r["image"]}') if c["type"] == "image" else L.Content.Text(c["text"]))
            kinds.append(c["type"])
        t1 = time.time(); txt = ask(eng, parts, r["messages"][0]["content"]); dt = time.time() - t1
        row = {"id": r["id"], "task": r["task"], "s": round(dt, 1), **score_analysis(r["meta"], txt), "out": txt}
        if cap["vision"] and i < n_gray:
            g = ask(eng, [L.Content.ImageFile(gray) if k == "image" else q for q, k in zip(parts, kinds)], r["messages"][0]["content"])
            row["gray_cats"] = score_analysis(r["meta"], g)["cats"]; row["gray_changed"] = set(row["gray_cats"]) != set(row["cats"])
        rows.append(row); REPORT[tag] = {**res, "rows": rows}; save()
        print(tag, r["id"], "valid", row["valid"], "agree", row["cat_agree"], f"{dt:.0f}s", flush=True)
    res["analysis"] = summarize(rows, ANALYSIS_KEYS); res["rows"] = rows
    g = [x["gray_changed"] for x in rows if "gray_changed" in x]
    res["image_ablation_changed"] = (sum(g) / len(g)) if g else None
    res["s_per_case_median"] = sorted(x["s"] for x in rows)[len(rows) // 2] if rows else None
    if cap["audio"]:
        try:
            import io

            import soundfile as sf
            from datasets import load_dataset
            ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
            a = ds[0]["audio"]; buf = io.BytesIO(); sf.write(buf, a["array"], a["sampling_rate"], format="WAV", subtype="PCM_16")
            res["audio_probe"] = {"reference": ds[0]["text"], "transcript": ask(eng,
                [L.Content.AudioBytes(buf.getvalue()), L.Content.Text("Transcribe exactly what is said. Output only the transcript.")], max_tok=120)}
        except Exception as e:  # noqa: BLE001
            res["audio_probe"] = f"error: {type(e).__name__}: {e}"
    del eng
    REPORT[tag] = res; save()
    print(tag, json.dumps({k: res[k] for k in ["load_s", "capabilities", "text_probe", "analysis", "image_ablation_changed", "s_per_case_median"]}), flush=True)
    return res


def module_diff():
    """Which parameter groups did the fine-tune change? (merged vs base, max |Δ|) — decides whether sections of the
    official .litertlm (vision/audio encoders) are valid for the fine-tuned model."""
    from safetensors import safe_open
    import torch
    base_dir = snapshot_download("unsloth/gemma-4-E2B-it", local_dir="/tmp/gemma4_e2b")
    def index(d):
        out = {}
        for f in glob.glob(f"{d}/*.safetensors"):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    out[k] = f
        return out
    bi, mi = index(base_dir), index(SRC)
    norm = lambda k: k.replace("base_model.model.", "")
    groups = {}
    for k, f in mi.items():
        kb = norm(k)
        if kb not in bi:
            groups.setdefault("_missing_in_base", []).append(kb); continue
        g = ("decoder_layers" if ".layers." in kb and "language_model" in kb else
             kb.split(".")[2] if kb.startswith("model.") and len(kb.split(".")) > 2 else kb.split(".")[0])
        with safe_open(f, "pt") as h1, safe_open(bi[kb], "pt") as h2:
            a, b = h1.get_tensor(k).float(), h2.get_tensor(kb).float()
        d = float((a - b).abs().max()) if a.shape == b.shape else float("inf")
        groups.setdefault(g, []).append(d)
    return {g: ({"n": len(v), "max_abs_diff": max(v), "changed": sum(x > 1e-3 for x in v)} if g != "_missing_in_base" else v[:20])
            for g, v in groups.items()}


REPORT["capabilities"] = caps(dst); save(); print("capabilities", REPORT["capabilities"], flush=True)
VAL_DIR = DATA
lval = {json.loads(line)["id"]: json.loads(line) for line in open(f"{DATA}/llm_val.jsonl")}
hf = None
if MODE == "merged" or MODE.startswith("lora"):
    try:
        REPORT["module_diff"] = module_diff(); save(); print("module_diff", json.dumps(REPORT["module_diff"])[:1500], flush=True)
    except Exception as e:  # noqa: BLE001
        REPORT["module_diff"] = f"error: {type(e).__name__}: {e}"
    hf = json.load(open(glob.glob("/kaggle/input/**/reports/full.json", recursive=True)[0]))
    # val records of THIS LoRA's SFT build (ids are reused across rebuilds): current dataset, else the frozen v1 copy
    cur = json.load(open(f"{DATA}/sft_data.json")).get("created_at")
    if hf.get("dataset_meta", {}).get("created_at") != cur:
        VAL_DIR = f"{DATA}/llm_eval"
        lval = {json.loads(line)["id"]: json.loads(line) for line in open(f"{VAL_DIR}/llm_val.jsonl")}
    REPORT["val_dir"] = VAL_DIR
    cases = [lval[x["id"]] for x in hf["L4_analysis_rows"]["E1"]]
else:
    cases = [r for r in lval.values() if r["task"] == "T1"][:10]
exp = run_suite(dst, "exported", cases, n_gray=5 if MODE == "stock" else 16)
if hf:   # ---- L6 export parity (test.md §5): HF merged (E1, greedy) vs this .litertlm (greedy), same prompts
    e1 = {x["id"]: x for x in hf["L4_analysis_rows"]["E1"]}
    a, b = summarize(list(e1.values()), ANALYSIS_KEYS), exp["analysis"]
    same_cats = sum(set(x["cats"]) == set(e1[x["id"]]["cats"]) for x in exp["rows"]) / len(exp["rows"])
    hf_abl = hf.get("image_ablation", {}).get("changed_rate")
    L6 = {"hf": a, "litertlm": b, "json_validity_drop": round(a["valid"] - b["valid"], 4), "category_agreement_drop": round(a["cat_agree"] - b["cat_agree"], 4),
          "category_sets_identical": round(same_cats, 4), "tier_ok": b["tier_ok"], "image_ablation_hf": hf_abl,
          "image_ablation_litertlm": exp["image_ablation_changed"], "starts_with_brace": sum(x["out"].lstrip().startswith("{") for x in exp["rows"]) / len(exp["rows"])}
    L6["pass"] = bool(L6["json_validity_drop"] <= 0.02 and L6["category_agreement_drop"] <= 0.03 and L6["tier_ok"] == 1.0
                      and (not hf_abl or (exp["image_ablation_changed"] or 0) >= 0.75 * hf_abl) and L6["starts_with_brace"] == 1.0)
    REPORT["L6"] = L6; save(); print("L6", json.dumps(L6), flush=True)
try:
    ref = hf_hub_download("litert-community/gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it.litertlm",
                          revision="b3ca0d2f076785a8f4b2219ddbd2bdb99954eae1", local_dir="/tmp/ref")
    run_suite(ref, "official_stock", cases[:10], n_gray=5)
except Exception as e:  # noqa: BLE001
    REPORT["official_stock"] = f"error: {type(e).__name__}: {e}"; save()
print("EXPORT_SUITE_DONE", flush=True)
