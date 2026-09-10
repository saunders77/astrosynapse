import math

import pytest
from astro2.experiment_control import certify, next_learning_rate


def rows(count, wins):
    return [{"pair": i, "scores": [float(i < wins)] * 2} for i in range(count)]


def test_certification_requires_seventy_percent_lower_bound_against_every_opponent():
    strong = rows(4096, 3277)
    borderline = rows(4096, 2868)
    assert certify({"a": strong, "b": strong}, pairs=4096, attempt=1)["passed"]
    result = certify({"a": strong, "b": borderline}, pairs=4096, attempt=1)
    assert not result["passed"]
    assert result["opponents"]["b"]["score"] >= 0.7
    assert result["opponents"]["b"]["lower"] < 0.7


def test_later_attempts_spend_less_error_and_partial_or_duplicate_pairs_cannot_certify():
    first = certify({"a": rows(100, 90)}, pairs=100, attempt=1)
    later = certify({"a": rows(100, 90)}, pairs=100, attempt=5)
    assert later["opponents"]["a"]["lower"] < first["opponents"]["a"]["lower"]
    assert sum(0.05 / (k * (k + 1)) for k in range(1, 10000)) < 0.05
    with pytest.raises(ValueError, match="complete predeclared"):
        certify({"a": rows(99, 90)}, pairs=100, attempt=1)
    duplicate = rows(100, 90)
    duplicate[-1]["pair"] = 0
    with pytest.raises(ValueError, match="unique"):
        certify({"a": duplicate}, pairs=100, attempt=1)


def test_controller_responds_to_measured_policy_change_and_obeys_bounds():
    assert next_learning_rate(1e-5, 1e-5, 2e-5, 0.01, 0.001, False) == pytest.approx(1.5e-5)
    assert next_learning_rate(2e-5, 1e-5, 2e-5, 0.01, 0.001, False) == 2e-5
    assert next_learning_rate(1e-5, 1e-5, 2e-5, 0.01, 0.03, False) < 1e-5
    assert next_learning_rate(1e-5, 1e-5, 2e-5, 0.01, 0.001, True) < 1e-5
    with pytest.raises(ValueError, match="invalid measured KL"):
        next_learning_rate(1e-5, 1e-5, 2e-5, 0.01, math.nan, False)
