"""NC7 regression tests: no ignored test suite, dashboard lineage.

Scope decision (CHANGELOG v1.3.0 Removed): the project was rebuilt to
zero derivatives, so the obsolete derivatives tests in
``tests/test_payoffs.py`` stay removed and no derivatives module is
restored. This unit finishes that removal by dropping the stale CI
ignore flag for the deleted file, and adds hashed lineage to the
dashboard generator so the numbers a generated page shows can be
traced to the config, data, and metrics files that produced them.

Expensive experiments are NOT re-executed here: only the offline fast
generation check runs (2 epochs, one floor, 64 paths). The full
dashboard training (40 epochs, three floors, 1000 paths), the NC5
risk-budgets study, and the tail-risk 38-window evaluation are not
independently reproduced by this unit.
"""

import glob
import hashlib
import json
import os
import re
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import make_dashboard


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def test_stage3_neural_cppi_7_nc7_ci_has_no_ignored_suite():
    """CI runs the whole suite with no ignore flag, and no derivatives
    import remains anywhere in the tree."""
    ci_path = os.path.join(REPO, ".github", "workflows", "ci.yml")
    ci_text = open(ci_path, encoding="utf-8").read()
    assert "--ignore" not in ci_text, "CI still ignores a test suite"
    assert not os.path.exists(os.path.join(REPO, "tests", "test_payoffs.py"))
    pattern = re.compile(r"(from|import)\s+dhl\.derivatives\b")
    offenders = []
    for root, _dirs, files in os.walk(REPO):
        if ".git" in root or ".venv" in root:
            continue
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            if os.path.abspath(path) == os.path.abspath(__file__):
                continue
            if pattern.search(open(path, encoding="utf-8").read()):
                offenders.append(os.path.relpath(path, REPO))
    assert offenders == [], f"derivatives imports remain: {offenders}"


def test_stage3_neural_cppi_7_nc7_lineage_matches_data_hash():
    """The lineage record hashes exactly the data files the demo
    training reads, and its config matches the fast build."""
    lineage = make_dashboard.dashboard_lineage(fast=True)
    assert lineage["generator"] == "scripts/make_dashboard.py"
    assert lineage["fast"] is True
    assert lineage["config"]["floors"] == [90.0]
    assert lineage["config"]["horizons"] == [30]
    assert lineage["config"]["multipliers"] == [4.0]
    assert lineage["config"]["n_epochs"] == 2
    assert lineage["config"]["n_paths"] == 64
    assert lineage["config"]["seeds"] == {
        "policy": 7, "simulation": 999, "multi": 11,
    }
    expected_files = sorted(
        os.path.relpath(p, REPO)
        for p in glob.glob(os.path.join(REPO, "data", "spy_daily_*.csv"))
    )
    assert sorted(lineage["data"]) == expected_files
    for rel, digest in lineage["data"].items():
        assert digest == _sha256(os.path.join(REPO, rel))
    for rel, digest in lineage["metrics"].items():
        path = os.path.join(REPO, rel)
        if os.path.exists(path):
            assert digest == _sha256(path)
        else:
            assert digest == "absent"
    commit = lineage["commit"]
    assert commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", commit)


def test_stage3_neural_cppi_7_nc7_wealth_chart_matches_source_metrics():
    """Numerical fixture: the wealth chart plots exactly the metrics it
    was handed, median and percentile band, for both arms."""
    days = np.arange(31, dtype=float)
    learned = {
        "p50": 100.0 + 0.1 * days,
        "p10": 99.0 + 0.05 * days,
        "p90": 101.0 + 0.15 * days,
    }
    classic = {
        "p50": 100.0 + 0.08 * days,
        "p10": 99.5 + 0.04 * days,
        "p90": 100.5 + 0.12 * days,
    }
    fig = make_dashboard.wealth_figure({(30, 4.0): {
        "learned": learned, "classic": classic,
    }})
    traces = list(fig.data)
    assert len(traces) == 4
    np.testing.assert_array_equal(
        np.asarray(traces[0].y, dtype=float), learned["p50"])
    np.testing.assert_array_equal(
        np.asarray(traces[1].y, dtype=float),
        np.concatenate([learned["p90"], learned["p10"][::-1]]))
    np.testing.assert_array_equal(
        np.asarray(traces[2].y, dtype=float), classic["p50"])
    np.testing.assert_array_equal(
        np.asarray(traces[3].y, dtype=float),
        np.concatenate([classic["p90"], classic["p10"][::-1]]))


def test_stage3_neural_cppi_7_nc7_dashboard_fast_generation(tmp_path):
    """Offline fast generation check: the dashboard builds offline and
    writes a lineage sidecar naming the exact inputs it used.

    The generated page itself is rebuilt by the presentation renderer,
    which drops HTML comments, so the lineage lives in the sidecar
    ``dashboard-lineage.json`` written next to the page, not in the
    page markup.
    """
    out = str(tmp_path / "dashboard.html")
    r = subprocess.run(
        [sys.executable, os.path.join("scripts", "make_dashboard.py"),
         "--fast", "--out", out],
        cwd=REPO, capture_output=True, text=True, timeout=1200,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert os.path.exists(out)
    assert os.path.getsize(out) > 100_000  # plotly.js is embedded
    html = open(out, encoding="utf-8").read()
    assert "Plotly.newPlot(" in html
    sidecar = os.path.join(str(tmp_path), "dashboard-lineage.json")
    assert os.path.exists(sidecar), "no lineage sidecar written"
    lineage = json.loads(open(sidecar, encoding="utf-8").read())
    assert lineage["fast"] is True
    assert lineage["config"]["n_epochs"] == 2
    spy_hash = _sha256(os.path.join(REPO, "data", "spy_daily_2007.csv"))
    assert lineage["data"]["data/spy_daily_2007.csv"] == spy_hash
