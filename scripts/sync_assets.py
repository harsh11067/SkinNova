"""Copy the single sources of truth into android assets (CLAUDE.md non-negotiable 7). Test: ml/tests/test_assets_sync.py."""
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
A = REPO / "android/app/src/main/assets"
PAIRS = [(p, A / "prompts" / p.name) for p in sorted((REPO / "ml/llm/prompts").glob("*.txt"))] + [
    (REPO / "ml/llm/cards/condition_cards.json", A / "prompts/condition_cards.json"),
    (REPO / "ml/llm/safety/rx_terms.txt", A / "safety/rx_terms.txt"),
    (REPO / "ml/llm/safety/intake_topics.json", A / "safety/intake_topics.json"),
    (REPO / "ml/common/labels.json", A / "labels.json"),
]

if __name__ == "__main__":
    for src, dst in PAIRS:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print(f"{src.relative_to(REPO)} -> {dst.relative_to(REPO)}")
    sys.exit(0)
