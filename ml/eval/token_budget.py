"""v2.1 item 4: can the engine's KV cache shrink from 4,096 to 2,048 tokens?

Counts tokens with the fine-tuned model's tokenizer (models/lora_v2/tokenizer.json) for every prompt the phone sends:
analysis (frozen llm_val, as rendered for the app), Ask SkinNova (chat_probe contexts × questions, + 2 earlier turns),
and translation (built from real model outputs). Budget = prompt + chat-template overhead + the task's max output.
Pre-registered rule (decisions 2026-10-09): 2,048 iff p99 of every budget ≤ 1,900.

  .venv/bin/python -m ml.eval.token_budget      → reports/token_budget.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from ml.common.paths import REPORTS, report_meta

REPO = Path(__file__).resolve().parents[2]
TOK = Tokenizer.from_file(str(Path.home() / "skinnova-data/models/lora_v2/tokenizer.json"))
TEMPLATE_OVERHEAD = 24          # <bos>, turn markers, roles, generation prompt (Gemma 4 chat template), generous
MAX_OUT = {"ANALYZE": 700, "REPAIR": 700, "CHAT": 300, "TRANSLATE": 600, "EXTRACT": 300}


def n(s: str) -> int:
    return len(TOK.encode(s, add_special_tokens=False).ids)


def pct(xs, q):
    return float(np.percentile(xs, q))


def main():
    recs = [json.loads(l) for l in open(REPO / "data/llm_eval/llm_val.jsonl")]
    analysis, extract, repair = [], [], []
    for r in recs:
        sys_msg = r["messages"][0]["content"]
        user = "".join(c["text"] for c in r["messages"][1]["content"] if c["type"] == "text")
        p = n(sys_msg) + n(user) + TEMPLATE_OVERHEAD
        if r["task"] in {"T1", "T9"}:
            analysis.append(p + MAX_OUT["ANALYZE"])
            # app: a NEW conversation = analysis system prompt + repair.txt (errors, schema, previous output ≤ 700 tokens)
            from ml.llm.prompt_builder import SCHEMA_COMPACT
            rep_t = (REPO / "ml/llm/prompts/repair.txt").read_text().split("\n", 1)[1].replace("{schema_compact}", SCHEMA_COMPACT)
            repair.append(n(sys_msg) + n(rep_t) + 700 + 60 + TEMPLATE_OVERHEAD + MAX_OUT["REPAIR"])
        elif r["task"] == "T6":
            extract.append(p + MAX_OUT["EXTRACT"])
    # Ask SkinNova: the chat probe's renderer, 2 earlier turns of ~60 tokens each
    from ml.eval import chat_probe as cp
    chat = []
    for ctx in cp.CONTEXTS:
        for lang, q, _ in cp.QUESTIONS:
            sysm, user = cp.render(ctx, q, lang, earlier=[("Is it contagious?", "x " * 60), ("What should I avoid?", "y " * 60)])
            chat.append(n(sysm) + n(user) + TEMPLATE_OVERHEAD + MAX_OUT["CHAT"])
    # translation: real outputs of the shipped model (select v2, no image) as the source JSON
    sel = json.loads((REPO / "reports/llm_litertlm_select_v2_noimage.json").read_text())
    tsys = (REPO / "ml/llm/prompts/translate_system.txt").read_text().split("\n", 1)[1]
    trans = [n(tsys) + n(row["out"]) + TEMPLATE_OVERHEAD + MAX_OUT["TRANSLATE"] for row in sel["rows"] if row.get("out")]
    budgets = {"analysis": analysis, "repair": repair, "extract": extract, "chat": chat, "translate": trans}
    rep = {**report_meta(), "tokenizer": "models/lora_v2/tokenizer.json", "template_overhead": TEMPLATE_OVERHEAD, "max_out": MAX_OUT,
           "budgets": {k: {"n": len(v), "p50": pct(v, 50), "p99": pct(v, 99), "max": max(v)} for k, v in budgets.items()}}
    worst = max(b["p99"] for b in rep["budgets"].values())
    rep["rule"] = "2048 iff every p99 <= 1900"; rep["worst_p99"] = worst; rep["fits_2048"] = worst <= 1900
    (REPORTS / "token_budget.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: {m: round(v[m]) for m in ("n", "p50", "p99", "max")} for k, v in rep["budgets"].items()}, indent=1))
    print("worst p99", worst, "fits 2048:", rep["fits_2048"])


if __name__ == "__main__":
    main()
