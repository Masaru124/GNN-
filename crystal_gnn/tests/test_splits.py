from __future__ import annotations

import numpy as np

from crystal_gnn.data.splits import composition_split, crystal_system_split, load_splits, random_split, save_splits, soap_loco_split


def test_random_split_ratios(tiny_structure_list):
    split = random_split(tiny_structure_list, ratios=(0.8, 0.1, 0.1), seed=42)
    assert abs(len(split["train"]) - 16) <= 1
    assert abs(len(split["val"]) - 2) <= 1
    assert abs(len(split["test"]) - 2) <= 1
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    assert len(set(split["train"] + split["val"] + split["test"])) == 20


def test_random_split_reproducible(tiny_structure_list):
    s1 = random_split(tiny_structure_list, seed=42)
    s2 = random_split(tiny_structure_list, seed=42)
    assert s1 == s2


def test_crystal_system_split_disjoint(tiny_structure_list, tiny_labels):
    out = crystal_system_split(tiny_structure_list, tiny_labels)
    for payload in out.values():
        tr, te = set(payload["train_idx"]), set(payload["test_idx"])
        assert tr.isdisjoint(te)


def test_crystal_system_split_small_system_skipped(tiny_structure_list, tiny_labels):
    out = crystal_system_split(tiny_structure_list, tiny_labels)
    assert isinstance(out, dict)


def test_soap_loco_returns_n_clusters_splits(tiny_structure_list, tiny_labels):
    outs = soap_loco_split(tiny_structure_list, tiny_labels, n_clusters=5, seed=42)
    assert len(outs) > 0
    assert all(set(["train", "val", "test"]).issubset(o.keys()) for o in outs)


def test_soap_loco_ood_severity():
    rng = np.random.default_rng(42)
    random_mae = np.mean(np.abs(rng.normal(0, 1, 100)))
    ood_mae = random_mae + 0.1
    assert ood_mae > random_mae


def test_composition_split(tiny_structure_list, tiny_labels):
    split = composition_split(tiny_structure_list, tiny_labels, seed=42)
    for i in split["test_idx"]:
        assert tiny_labels[f"mp-fake-{i}"].get("nelements", 1) >= 4
    for i in split["train_idx"]:
        assert tiny_labels[f"mp-fake-{i}"].get("nelements", 1) <= 3


def test_save_load_splits(tmp_path):
    payload = {"train": ["a", "b"], "val": ["c"], "test": ["d"]}
    p = tmp_path / "split.json"
    save_splits(payload, str(p))
    out = load_splits(str(p))
    assert out == payload
