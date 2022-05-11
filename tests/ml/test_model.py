
from starter.ml.model import compute_model_metrics
import pytest

def test_compute_model_metrics():
    y = [0, 0, 0, 1, 1, 0, 0,0, 0, 1, 0, 1, 1, 1, 0]
    preds = [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
    expected_metrics = [0.6, 1.0, 0.74999]
    computed_metrics = compute_model_metrics(y, preds)
    assert computed_metrics == pytest.approx(expected_metrics, 0.001)