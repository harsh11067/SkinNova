"""Voice intake evaluation on the recorded voice_test set (test.md §9 V2–V6).

Layout (Harsh records it; protocol and per-speaker schedule in docs/voice_script_sheet.md):
  data/voice_test/clips/<file>.wav          16 kHz mono PCM preferred
  data/voice_test/manifest.csv              clip,speaker,lang,noise,card_id,mode,omitted,reference_transcript
    lang = en | hi | hinglish  · noise = quiet | noisy · mode = free | read
    omitted = card fields the speaker did NOT say (";"-separated), filled in while typing reference_transcript
    reference_transcript = what was actually said, typed by a person (hi in Devanagari, hinglish in Roman)

Transcripts (V2) come from either
  --transcripts F.jsonl   {"clip": ..., "transcript": ...} per line (e.g. pulled from the phone), or
  --model X.litertlm      transcribe on the PC with LiteRT-LM 0.17.1 (needs an audio-capable model) using the app's
                          transcribe prompt (PromptBuilder.transcribe / SessionViewModel).
Extraction (V3–V6) always runs `--model` on the transcript with the app's extract prompt, then the same IntakeValidator
as the app (evidence quotes must be in the transcript). Runs in .venv-export:
  .venv-export/bin/python -m ml.eval.eval_voice --model models/litertlm/stock/gemma-4-E2B-it.litertlm --tag stock
Writes reports/voice_eval_<tag>.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path

from ml.common.paths import DATA, REPO, REPORTS, report_meta
from ml.llm.prompt_builder import PROMPTS, build_extract, load_template, runtime_system_message
from ml.voice.intake import FIELDS, lev, validate_intake

VOICE = DATA / "voice_test"
CARDS = REPO / "ml/voice/voice_cards.json"
LANGS, NOISES = ("en", "hi", "hinglish"), ("quiet", "noisy")
TRANSCRIBE_SYSTEM = "You transcribe speech exactly."          # SessionViewModel.startRecording
LANG_HINT = {"en": "English or Hindi", "hi": "Hindi or Hinglish", "hinglish": "Hindi or Hinglish"}   # app setting en / hi


def load_cards(path: Path = CARDS) -> dict:
    return {c["id"]: c for c in json.loads(path.read_text())["cards"]}


def load_manifest(path: Path) -> list[dict]:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        if r["lang"] not in LANGS or r["noise"] not in NOISES:
            raise ValueError(f"{r['clip']}: lang must be {LANGS}, noise {NOISES}")
    return rows


def gold_for(row: dict, cards: dict) -> dict:
    """Card ground truth minus the facts this speaker did not actually say."""
    card = cards[row["card_id"]]
    omitted = {x.strip() for x in (row.get("omitted") or "").split(";") if x.strip()}
    unknown = omitted - set(card["fields"])
    if unknown:
        raise ValueError(f"{row['clip']}: omitted {sorted(unknown)} are not fields of {row['card_id']}")
    return {k: v for k, v in card["fields"].items() if k not in omitted}


def norm_text(s: str) -> str:
    """NFC, lowercase, punctuation/symbols removed (Devanagari vowel signs and virama are kept), spaces collapsed."""
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(" " if unicodedata.category(ch)[0] in "PS" else ch for ch in s)
    return " ".join(s.split())


def cer(hyp: str, ref: str) -> float:
    h, r = norm_text(hyp), norm_text(ref)
    return lev(h, r) / max(1, len(r))


def script_of(s: str) -> str:
    deva = sum(1 for ch in s if "ऀ" <= ch <= "ॿ")
    latn = sum(1 for ch in s if ch.isascii() and ch.isalpha())
    return "deva" if deva > latn else "latn"


def score_clip(row: dict, card: dict, gold: dict, transcript: str, extraction: str) -> dict:
    v = validate_intake(extraction, transcript)
    pred = {k: x["value"] for k, x in v["fields"].items() if x is not None}
    ref = row.get("reference_transcript", "")
    same_script = bool(ref) and script_of(ref) == script_of(transcript)
    neg = [k for k in card["negated"] if k in gold]
    return {"clip": row["clip"], "speaker": row["speaker"], "lang": row["lang"], "noise": row["noise"], "card": row["card_id"],
            "mode": row.get("mode", ""), "valid": v["ok"],
            "cer": round(cer(transcript, ref), 4) if same_script else None, "script_mismatch": bool(ref) and not same_script,
            "n_gold": len(gold), "correct": sum(pred.get(k) == g for k, g in gold.items()),
            "wrong": sorted(k for k in gold if k in pred and pred[k] != gold[k]), "missed": sorted(k for k in gold if k not in pred),
            "hallucinated": sorted(k for k in pred if k not in gold),
            "neg_n": len(neg), "neg_correct": sum(pred.get(k) == gold[k] for k in neg),
            "dur_numeral": bool(card["numeral"]) and "duration" in gold,
            "dur_correct": ("duration" in gold and pred.get("duration") == gold["duration"]),
            "dropped": v["dropped"], "transcript": transcript[:400], "extraction": extraction[:600]}


def _rate(num: int, den: int):
    return round(num / den, 4) if den else None


def summarize(rows: list[dict]) -> dict:
    cers = defaultdict(list)
    for r in rows:
        if r["cer"] is not None:
            cers[(r["lang"], r["noise"])].append(r["cer"])
    by_lang = {}
    for lang in LANGS:
        sub = [r for r in rows if r["lang"] == lang]
        by_lang[lang] = {"n": len(sub), "field_acc": _rate(sum(r["correct"] for r in sub), sum(r["n_gold"] for r in sub))}
    dur = [r for r in rows if r["dur_numeral"]]
    v3_ok = all(x["field_acc"] is not None and x["field_acc"] >= 0.85 for x in by_lang.values())
    halluc = sum(len(r["hallucinated"]) for r in rows)
    neg = _rate(sum(r["neg_correct"] for r in rows), sum(r["neg_n"] for r in rows))
    v6 = _rate(sum(r["dur_correct"] for r in dur), len(dur))
    quiet_cer = {lang: (round(statistics.mean(cers[(lang, "quiet")]), 4) if cers[(lang, "quiet")] else None) for lang in ("en", "hi")}
    return {
        "n_clips": len(rows), "speakers": len({r["speaker"] for r in rows}), "json_valid": _rate(sum(r["valid"] for r in rows), len(rows)),
        "V2_cer": {f"{lang}/{noise}": {"n": len(cers[(lang, noise)]), "mean": round(statistics.mean(cers[(lang, noise)]), 4),
                                       "median": round(statistics.median(cers[(lang, noise)]), 4)}
                   for lang in LANGS for noise in NOISES if cers[(lang, noise)]},
        "V2_script_mismatch": sum(r["script_mismatch"] for r in rows),
        "V2_ship_non_beta": all(c is not None and c <= 0.25 for c in quiet_cer.values()),
        "V3_field_acc": by_lang, "V3_pass": v3_ok,
        "V4_hallucinated_fields": halluc, "V4_pass": halluc == 0,
        "V5_negation_acc": neg, "V5_pass": neg is not None and neg >= 0.90,
        "V6_duration_acc": v6, "V6_pass": v6 is not None and v6 >= 0.90,
    }


def coverage_problems(rows: list[dict]) -> list[str]:
    """test.md §1: ≥ 120 clips, ≥ 6 speakers, every lang × noise cell present."""
    out = []
    if len(rows) < 120:
        out.append(f"{len(rows)} clips < 120")
    if len({r['speaker'] for r in rows}) < 6:
        out.append("fewer than 6 speakers")
    cells = {(r["lang"], r["noise"]) for r in rows}
    out += [f"no clips for {lang}/{noise}" for lang in LANGS for noise in NOISES if (lang, noise) not in cells]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".litertlm used for extraction (and transcription unless --transcripts)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--manifest", default=str(VOICE / "manifest.csv"))
    ap.add_argument("--transcripts", help="jsonl {clip, transcript}; skips on-PC transcription")
    a = ap.parse_args()
    import litert_lm as L

    cards, rows_in = load_cards(), load_manifest(Path(a.manifest))
    given = {}
    if a.transcripts:
        given = {d["clip"]: d["transcript"] for d in map(json.loads, open(a.transcripts, encoding="utf-8"))}
    caps = L.Capabilities(a.model); m = caps.input_modalities; has_audio = bool(m.audio); caps.close()
    if not a.transcripts and not has_audio:
        raise SystemExit(f"{a.model} has no audio encoder: pass --transcripts (e.g. from the phone's on-device recogniser)")
    eng = L.Engine(a.model, backend=L.Backend.CPU(), audio_backend=L.Backend.CPU() if has_audio and not a.transcripts else None,
                   max_num_tokens=4096)
    _, transcribe_t = load_template("transcribe.txt", PROMPTS)

    def ask(parts, system, max_tok):
        conv = eng.create_conversation(system_message=runtime_system_message(system), sampler_config=L.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0, seed=3407),
                                       thinking_config=L.ThinkingConfig(enable_thinking=False), max_output_tokens=max_tok)
        try:
            out = conv.send_message(L.Contents(parts))
        finally:
            conv.close()
        return "".join(c.get("text", "") for c in out.get("content", []) if isinstance(c, dict)).strip()

    rows = []
    for r in rows_in:
        if a.transcripts:
            transcript = given.get(r["clip"])
            if transcript is None:
                raise SystemExit(f"no transcript for {r['clip']} in {a.transcripts}")
        else:
            wav = (VOICE / "clips" / r["clip"]).resolve()
            transcript = ask([L.Content.AudioFile(str(wav)), L.Content.Text(transcribe_t.replace("{lang_hint}", LANG_HINT[r["lang"]]))],
                             TRANSCRIBE_SYSTEM, 200)
        p = build_extract(transcript)
        extraction = ask([L.Content.Text(p["user"])], p["system"], 300)
        rows.append(score_clip(r, cards[r["card_id"]], gold_for(r, cards), transcript, extraction))
        print(r["clip"], "cer", rows[-1]["cer"], "fields", f'{rows[-1]["correct"]}/{rows[-1]["n_gold"]}', "halluc", rows[-1]["hallucinated"], flush=True)
    rep = {**report_meta(), "tag": a.tag, "model": Path(a.model).name, "transcripts": a.transcripts or "litert-lm audio (PC)",
           "coverage_problems": coverage_problems(rows), "fields": list(FIELDS), **summarize(rows), "rows": rows}
    out = REPORTS / f"voice_eval_{a.tag}.json"
    out.write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
