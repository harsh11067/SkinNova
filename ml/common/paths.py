"""Central paths + env loading. Every script imports from here so locations live in one place."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements, but keep pure-python tests runnable
    load_dotenv = None

REPO = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(REPO / ".env")

DATA = Path(os.environ.get("SKINNOVA_DATA_DIR") or (REPO / "data")).expanduser()
MODELS = Path(os.environ.get("SKINNOVA_MODELS_DIR") or (REPO / "models")).expanduser()
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
IMAGES = PROCESSED / "images"
SPLITS = DATA / "splits"
LLM_DATA = DATA / "llm"
REPORTS = REPO / "reports"
FIXTURES = REPO / "tests" / "fixtures"
ANDROID_ASSETS = REPO / "android" / "app" / "src" / "main" / "assets"
PROMPTS = REPO / "ml" / "llm" / "prompts"
CARDS = REPO / "ml" / "llm" / "cards" / "condition_cards.json"
LABELS = REPO / "ml" / "common" / "labels.json"

SEED = int(os.environ.get("SEED", "3407"))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def report_meta(dataset_rev: str = "", model_sha: str = "") -> dict:
    """Every report JSON carries these (CLAUDE.md non-negotiable 8)."""
    import datetime as _dt
    return {"git_sha": git_sha(), "dataset_rev": dataset_rev, "model_sha": model_sha,
            "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")}
