"""Publication figures must be independent of ambient style and call order."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pytest

from src import render_conjunction_mechanism as mechanism
from src import render_conjunction_robustness as robustness

ROOT = Path(__file__).resolve().parents[1]


def pixels(folder):
    return {p.name: mpimg.imread(p) for p in folder.glob("*.png")}


def test_render_order_and_ambient_style_do_not_change_figures(tmp_path, monkeypatch):
    monkeypatch.setattr(mechanism, "REPORTS", ROOT / "artifacts/expected/conjunction_mechanism")
    monkeypatch.setattr(mechanism, "FIGURES", tmp_path / "mechanism-first")
    source = ROOT / "artifacts/expected/conjunction_robustness"
    with plt.rc_context():
        plt.rcdefaults()
        initial = {k: plt.rcParams[k] for k in ("text.color", "axes.labelcolor", "font.size", "axes.facecolor")}
        mechanism.main(figures=tmp_path / "mechanism-first")
        robustness.render(source, tmp_path / "robustness-second")
        assert {k: plt.rcParams[k] for k in initial} == initial
        plt.rcParams.update({"text.color": "red", "axes.labelcolor": "green",
                             "font.size": 23, "axes.facecolor": "pink"})
        changed = {k: plt.rcParams[k] for k in initial}
        robustness.render(source, tmp_path / "robustness-first")
        mechanism.main(figures=tmp_path / "mechanism-second")
        assert {k: plt.rcParams[k] for k in initial} == changed
    for first, second in [("mechanism-first", "mechanism-second"),
                          ("robustness-first", "robustness-second")]:
        a, b = pixels(tmp_path / first), pixels(tmp_path / second)
        assert a.keys() == b.keys() and a
        for name in a:
            np.testing.assert_array_equal(a[name], b[name], err_msg=name)


def test_style_is_restored_when_rendering_fails(tmp_path, monkeypatch):
    def fail():
        raise ValueError("test source failure")
    monkeypatch.setattr(mechanism, "check_sources", fail)
    with plt.rc_context({"text.color": "red", "font.size": 27}):
        with pytest.raises(ValueError, match="test source failure"):
            mechanism.main()
        assert plt.rcParams["text.color"] == "red"
        assert plt.rcParams["font.size"] == 27
