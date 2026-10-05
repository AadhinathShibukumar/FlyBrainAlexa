import numpy as np
import pandas as pd

from src.brain.build_functional_groups import build_groups


def test_groups_use_annotation_rows_not_index_fallbacks() -> None:
    annotations = pd.DataFrame(
        {
            "root_id": [10, 20, 30, 40, 50, 60],
            "super_class": [
                "central",
                "sensory",
                "ascending",
                "descending",
                "motor",
                "central",
            ],
            "cell_class": [
                "olfactory",
                "mechanosensory",
                "visual",
                "unknown",
                "unknown",
                "Kenyon_Cell",
            ],
            "flow": ["afferent", "afferent", "afferent", "efferent", "efferent", "intrinsic"],
            "side": ["left", "right", "left", "left", "right", "center"],
        }
    )
    groups = build_groups(annotations, np.array([60, 50, 40, 30, 20, 10]))
    assert groups["olfactory"].tolist() == [5]
    assert groups["descending_motor"].tolist() == [2]
    assert groups["motor"].tolist() == [1]
