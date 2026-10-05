"""Timeline maths (TL3 building blocks) on shared fixtures; TL1 gates read from reports/timeline_eval.json."""
import json

import numpy as np
import pytest

from ml.common.paths import FIXTURES, REPORTS
from ml.timeline.colour import ciede2000, srgb_to_lab
from ml.timeline.metrics import js_divergence


def test_ciede2000_sharma():
    d = json.loads((FIXTURES / "ciede2000_cases.json").read_text())
    assert len(d["pairs"]) == 34
    for p in d["pairs"]:
        assert ciede2000(p[:3], p[3:6]) == pytest.approx(p[6], abs=1e-4)


def test_lab_reference_points():
    assert srgb_to_lab(np.array([255, 255, 255])) == pytest.approx([100, 0, 0], abs=1e-2)
    assert srgb_to_lab(np.array([0, 0, 0])) == pytest.approx([0, 0, 0], abs=1e-6)
    assert srgb_to_lab(np.array([255, 0, 0])) == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


def test_js_bounds():
    assert js_divergence([.5, .5], [.5, .5]) == pytest.approx(0, abs=1e-9)
    assert js_divergence([1, 0], [0, 1]) == pytest.approx(1, abs=1e-6)


@pytest.mark.skipif(not (REPORTS / "timeline_eval.json").exists(), reason="TL1 eval not run")
def test_tl1_report_present_and_honest():
    r = json.loads((REPORTS / "timeline_eval.json").read_text())
    assert r["n_pairs"] >= 250 and "gates" in r          # gates are reported, pass or fail; the app labels beta on fail
