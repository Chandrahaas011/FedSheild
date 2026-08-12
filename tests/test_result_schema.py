import pytest

from src.evaluation.result_schema import validate_result


def test_result_schema_rejects_incomplete_records():
    with pytest.raises(ValueError):
        validate_result({"status": "complete"})

