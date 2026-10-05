"""Non-negotiable 7: training prompts ≡ app prompts; cards, rx terms, labels identical in android assets."""
import filecmp
import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("sync_assets", REPO / "scripts/sync_assets.py")
sync = importlib.util.module_from_spec(spec); spec.loader.exec_module(sync)


def test_assets_identical():
    for src, dst in sync.PAIRS:
        assert dst.exists(), f"missing {dst} — run scripts/sync_assets.py"
        assert filecmp.cmp(src, dst, shallow=False), f"{dst} differs from {src}"


def test_every_prompt_has_version_header():
    for p in (REPO / "ml/llm/prompts").glob("*.txt"):
        assert p.read_text().startswith("# v1\n"), p
