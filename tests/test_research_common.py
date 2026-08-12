import pandas as pd

from experiments.research_common import stratified_cap


def test_stratified_cap_is_deterministic_and_retains_rare_family():
    frame = pd.DataFrame(
        {
            "family_label": [0] * 100 + [1] * 20 + [2] * 2,
            "value": list(range(122)),
        }
    )
    first = stratified_cap(frame, 30, seed=3)
    second = stratified_cap(frame, 30, seed=3)
    assert first.equals(second)
    assert len(first) == 30
    assert set(first["family_label"]) == {0, 1, 2}

