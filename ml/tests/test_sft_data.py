"""L0 — SFT dataset validation (test.md §5). Skipped until ml/llm/build_sft_dataset.py has run."""
import json
from collections import Counter

import pandas as pd
import pytest

from ml.common.paths import LLM_DATA, REPO, REPORTS, SPLITS
from ml.common.schema import TIER_RANK, label_keys
from ml.llm import prompt_builder as pb
from ml.llm.validate import load_rx_terms, validate
from ml.voice.intake import evidence_ok, validate_intake

HAVE = (LLM_DATA / "sft_train.jsonl").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="SFT data not built")
KEYS = label_keys()
RX = load_rx_terms()


def load(name):
    return [json.loads(l) for l in open(LLM_DATA / f"{name}.jsonl")]


@pytest.fixture(scope="module")
def train():
    return load("sft_train")


def test_every_analysis_target_validates(train):
    for r in train + load("sft_val") + load("llm_val") + load("llm_test"):
        if r["task"] in {"T1", "T2", "T3", "T4", "T5", "T9"}:
            m = r["meta"]
            txt = r["messages"][-1]["content"][0]["text"]
            v = validate(txt, KEYS, cv_top1_p=m["cv_top1_p"], rule_tier=m["rule_tier"], rx_terms=RX)
            assert v.ok, (r["id"], v.errors)
            assert TIER_RANK[json.loads(txt)["triage"]["tier"]] >= TIER_RANK[m["rule_tier"]]


def test_rendered_with_shipped_prompts(train):
    """System prompts are byte-identical to the app's asset templates (non-negotiable 7)."""
    shipped = {n: pb.load_template(n, REPO / "android/app/src/main/assets/prompts")[1]
               for n in ["analyze_system.txt", "extract_system.txt", "narrate_system.txt"]}
    for r in train:
        sys_txt = r["messages"][0]["content"]
        if r["task"] in {"T1", "T2", "T3", "T4", "T5", "T9"}:
            assert sys_txt == shipped["analyze_system.txt"]
        elif r["task"] == "T6":
            assert sys_txt == shipped["extract_system.txt"]
        elif r["task"] == "T7":
            assert sys_txt == shipped["narrate_system.txt"]


def test_image_before_text(train):
    for r in train:
        content = r["messages"][1]["content"]
        types = [c["type"] for c in content]
        if "image" in types:
            assert types[0] == "image" and "image" in r, r["id"]
            assert (LLM_DATA / r["image"]).exists(), r["image"]


def test_no_eval_images_in_sft_train(train):
    """Leakage: SFT images come only from the train split (or synthetic non-skin images)."""
    held = set()
    for s in ["val", "test", "external_test"]:
        held |= {p.split("/")[-1] for p in pd.read_csv(SPLITS / f"{s}.csv").img}
    used = {r["image"].split("/")[-1] for r in train if "image" in r}
    assert not (used & held)


def test_task_mix_and_class_cap(train):
    rep = json.loads((REPORTS / "sft_data.json").read_text())
    mix = Counter(r["task"] for r in train); n = sum(mix.values())
    for t, share in rep["task_mix_target"].items():
        assert abs(mix[t] / n - share) <= 0.02, (t, mix[t] / n, share)
    t1 = Counter(r["meta"]["label"] for r in train if r["task"] == "T1")
    assert max(t1.values()) / sum(t1.values()) <= 0.25, t1


def test_extraction_evidence_in_transcript(train):
    for r in train + load("llm_val"):
        if r["task"] == "T6":
            tr = r["meta"]["transcript"]
            out = json.loads(r["messages"][-1]["content"][0]["text"])
            for f, item in out["fields"].items():
                if item is not None:
                    assert evidence_ok(item["evidence"], tr), (r["id"], f, item)
            v = validate_intake(r["messages"][-1]["content"][0]["text"], tr)
            assert set(k for k, x in v["fields"].items() if x) == set(r["meta"]["fields"]), r["id"]


def test_heldout_phrasing_not_in_training(train):
    """T6 eval transcripts use phrasings reserved from training (generalisation, not memorisation)."""
    from ml.llm.phrasebank import INTAKE
    held = {opts[-1] for f in INTAKE.values() for v in f.values() for opts in v.values() if len(opts) > 1}
    for r in train:
        if r["task"] == "T6":
            for f, item in json.loads(r["messages"][-1]["content"][0]["text"])["fields"].items():
                if item is not None:
                    assert item["evidence"] not in held, (r["id"], item["evidence"])
