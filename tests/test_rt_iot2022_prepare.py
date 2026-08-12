import numpy as np
import pandas as pd

from src.data.prepare_rt_iot2022 import BENIGN_LABELS


def test_rt_iot2022_benign_mapping_is_explicit():
    assert BENIGN_LABELS == {"Thing_Speak", "MQTT_Publish", "Wipro_bulb"}


def test_feature_fingerprints_identify_equal_rows():
    frame = pd.DataFrame({"a": np.array([1, 1, 2], dtype=np.float32), "b": [3.0, 3.0, 4.0]})
    hashes = pd.util.hash_pandas_object(frame, index=False)
    assert hashes.iloc[0] == hashes.iloc[1]
    assert hashes.iloc[0] != hashes.iloc[2]
