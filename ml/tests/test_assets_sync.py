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
    """Every prompt carries a version header. The TRAINED prompts stay at v1 (the LoRA learned them); the Ask SkinNova
    prompts (chat_*.txt) are app-only (never trained) and versioned on their own (v2: grounded in the care notes)."""
    import re
    for p in (REPO / "ml/llm/prompts").glob("*.txt"):
        head = p.read_text().split("\n", 1)[0]
        assert re.fullmatch(r"# v\d+", head), p
        if not p.name.startswith("chat_"):
            assert head == "# v1", f"{p}: trained prompt changed version"
