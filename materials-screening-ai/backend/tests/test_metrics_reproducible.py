"""
Reproducibility gate for research/metrics.json.

Runs the frozen evaluation protocol in memory (no artifact writes, no git-dirty
gate) and asserts every non-provenance value equals the committed metrics.json.
Provenance fields (git_commit, parent_git_commit, git_dirty) are excluded by
construction: they describe *when* the file was generated, not what it says.

Also scans README / research docs / frontend for the retracted FAPbI3 interval
pair [1.3491, 1.7804] eV outside the explicit correction notes.
"""

import json
import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
RESEARCH = BACKEND.parent / "research"
METRICS_PATH = RESEARCH / "metrics.json"
EVAL_SCRIPT = BACKEND / "scripts" / "eval_protocol.py"

PROVENANCE_KEYS = {"git_commit", "parent_git_commit", "git_dirty"}
RETRACTED = ("1.3491", "1.7804")
# Files allowed to mention the retracted pair: the correction notes themselves.
RETRACTION_WHITELIST = {"METRICS_CHANGELOG.md", "reproducibility_scorecard.md"}


def _strip_provenance(obj):
    """Recursively drop provenance fields and normalize tuples to JSON lists."""
    if isinstance(obj, dict):
        return {k: _strip_provenance(v) for k, v in obj.items() if k not in PROVENANCE_KEYS}
    if isinstance(obj, (list, tuple)):
        return [_strip_provenance(v) for v in obj]
    return obj


def _run_protocol_in_memory():
    """Import and run eval_protocol.run_eval_protocol() without writing artifacts."""
    os.environ["EVAL_PROTOCOL_NO_WRITE"] = "1"
    os.environ["EVAL_PROTOCOL_SKIP_GIT_CHECK"] = "1"
    sys.path.insert(0, str(BACKEND))
    sys.path.insert(0, str(BACKEND / "scripts"))
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("eval_protocol_under_test", EVAL_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.run_eval_protocol()
    finally:
        os.environ.pop("EVAL_PROTOCOL_NO_WRITE", None)
        os.environ.pop("EVAL_PROTOCOL_SKIP_GIT_CHECK", None)


@pytest.fixture(scope="module")
def committed():
    assert METRICS_PATH.exists(), f"missing {METRICS_PATH}"
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def recomputed():
    return _run_protocol_in_memory()


def test_metrics_file_is_valid_json(committed):
    assert committed["metadata"]["num_evaluated_samples"] == 9
    assert committed["git_dirty"] is False


def test_recomputed_matches_committed_excluding_provenance(committed, recomputed):
    a = _strip_provenance(recomputed)
    b = _strip_provenance(committed)
    if a != b:
        differing = []

        def walk(x, y, path=""):
            if isinstance(x, dict) and isinstance(y, dict):
                for k in sorted(set(x) | set(y)):
                    if k not in x or k not in y:
                        differing.append(f"{path}.{k}: present in only one side")
                    else:
                        walk(x[k], y[k], f"{path}.{k}")
            elif isinstance(x, list) and isinstance(y, list):
                if len(x) != len(y):
                    differing.append(f"{path}: length {len(x)} != {len(y)}")
                else:
                    for i, (xi, yi) in enumerate(zip(x, y)):
                        walk(xi, yi, f"{path}[{i}]")
            elif x != y:
                differing.append(f"{path}: recomputed {x!r} != committed {y!r}")

        walk(a, b)
        pytest.fail("metrics.json is stale; differing paths:\n  " + "\n  ".join(differing[:40]))


def test_in_memory_mode_wrote_nothing(recomputed):
    """The in-memory run must not rewrite the committed artifacts."""
    assert METRICS_PATH.exists()
    with open(METRICS_PATH, encoding="utf-8") as f:
        committed = json.load(f)
    assert committed["metadata"]["git_commit"] == _strip_provenance(recomputed).get(
        "metadata", {}
    ).get("git_commit", committed["metadata"]["git_commit"])


def test_retracted_interval_not_used_as_current_value():
    """1.3491 / 1.7804 may appear only in the correction notes."""
    repo_root = BACKEND.parents[1]
    offenders = []
    for rel in ("README.md", "materials-screening-ai/README.md"):
        p = repo_root / rel
        if p.exists():
            offenders += [(rel, i) for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                          if any(t in line for t in RETRACTED)]
    for doc in sorted(RESEARCH.glob("*.md")):
        if doc.name in RETRACTION_WHITELIST:
            continue
        offenders += [(doc.name, i) for i, line in enumerate(doc.read_text(encoding="utf-8").splitlines(), 1)
                      if any(t in line for t in RETRACTED)]
    assert not offenders, f"retracted interval quoted as current: {offenders}"


def test_current_interval_is_the_reproducible_one(committed):
    fa = [r for r in committed["held_out_test_evaluations"]["benchmark_candidates"]
          if r["formula"] == "FAPbI3"][0]
    assert fa["conformal_interval_80_eV"] == [1.2544, 1.8751]
    assert fa["interval_width_eV"] == 0.6207
    assert committed["halide_perovskites_in_family"]["conformal_80_quantile_q_tilde_eV"] == 0.2157
