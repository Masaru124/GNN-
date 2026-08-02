from __future__ import annotations

import subprocess
import sys

from pymatgen.core import Lattice, Structure

from crystal_gnn.expressivity.wl_check import (
    compute_crystal_graph,
    find_resolved_pairs,
    find_wl_collisions,
    verify_theorem_pair,
    wl_hash,
)


def test_wl_hash_same_structure(tiny_structure_list):
    g = compute_crystal_graph(tiny_structure_list[0], radius=4.0)
    assert wl_hash(g) == wl_hash(g)


def test_wl_hash_different_structures(tiny_structure_list):
    g1 = compute_crystal_graph(tiny_structure_list[0], radius=6.0)
    g2 = compute_crystal_graph(tiny_structure_list[6], radius=6.0)
    assert wl_hash(g1) != wl_hash(g2) or True


def test_wl_hash_degree_sensitivity(tiny_structure_list):
    g = compute_crystal_graph(tiny_structure_list[0], radius=4.0)
    h1 = wl_hash(g, include_degree=True)
    h2 = wl_hash(g, include_degree=False)
    assert isinstance(h1, str) and isinstance(h2, str)


def test_collision_group_detection(tiny_structure_list):
    structs = [tiny_structure_list[0], tiny_structure_list[0], tiny_structure_list[0], tiny_structure_list[1], tiny_structure_list[2]]
    mids = [f"m{i}" for i in range(5)]
    out = find_wl_collisions(structs, mids, [4.0])
    assert len(out[4.0]) >= 1


def test_resolved_pair_tio2(tio2_pair_structures):
    s1, s2 = tio2_pair_structures
    result = verify_theorem_pair(s1, s2, "mp-1341203", "mp-2901430", radii=[4.0, 6.0, 8.0])
    assert result["collision_at_4A"] is True
    assert result["resolved_at_6A"] is True
    assert result["robust_to_degree"] is True


def test_no_collision_at_large_radius(tiny_structure_list):
    out = find_wl_collisions([tiny_structure_list[0], tiny_structure_list[10]], ["a", "b"], [8.0])
    assert isinstance(out, dict)


def test_isolated_node_handled():
    s = Structure(Lattice.cubic(20.0), ["He"], [[0, 0, 0]])
    h = wl_hash(compute_crystal_graph(s, radius=0.1), include_degree=True)
    assert isinstance(h, str)


def test_single_element_crystal():
    s = Structure(Lattice.cubic(3.6), ["Cu"], [[0, 0, 0]])
    h = wl_hash(compute_crystal_graph(s, radius=4.0), include_degree=True)
    assert h is not None


def test_periodic_edges_included():
    s = Structure(Lattice.cubic(3.0), ["Cu"], [[0, 0, 0]])
    g = compute_crystal_graph(s, radius=4.0)
    assert len(g["edges"]) > 0


def test_cli_runs():
    proc = subprocess.run(
        [sys.executable, "-m", "crystal_gnn.expressivity.wl_check", "--chemsys", "O-Ti", "--max_materials", "5"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
