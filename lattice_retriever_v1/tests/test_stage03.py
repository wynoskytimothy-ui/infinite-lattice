"""Stage 03 gate — frequency → rotation / quadrant."""

from lattice_retriever_v1.stage03_rotation import rotate_token, wing_from_frequency_profile


def test_wing_stable():
    dfs = (100, 50, 10)
    assert wing_from_frequency_profile(dfs) == wing_from_frequency_profile(dfs)


def test_anagram_order_changes_quadrant():
    """tas vs sat — same letter multiset, different order → different wing."""
    tas = rotate_token("tas", {"t": 500, "a": 200, "s": 50})
    sat = rotate_token("sat", {"s": 50, "a": 200, "t": 500})
    assert tas.quadrant != sat.quadrant
    assert tas.frequency_profile != sat.frequency_profile


def test_task_vs_sate_different_quadrants():
    task = rotate_token(
        "task",
        {"t": 400, "a": 120, "s": 80, "k": 900},
    )
    sate = rotate_token(
        "sate",
        {"s": 900, "a": 120, "t": 80, "e": 400},
    )
    assert task.quadrant != sate.quadrant


def test_rotation_glass_box():
    r = rotate_token("ing", {"i": 300, "n": 150, "g": 40})
    ex = r.explain()
    assert ex["text"] == "ing"
    assert 1 <= ex["quadrant"] <= 32
    assert len(ex["frequency_profile"]) == 3
    assert "cell" in ex
