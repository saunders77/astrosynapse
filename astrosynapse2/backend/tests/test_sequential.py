import numpy as np
import pytest
from astro2.sequential import log_evidence, lower_sequence, promotion_evidence


def test_constant_tie_never_proves_improvement_and_larger_samples_tighten_bound():
    assert log_evidence([0.5] * 10000, 0.5) == 0
    assert lower_sequence([0.5] * 1000) < 0.5
    assert lower_sequence([0.55] * 1000) > 0.5
    assert lower_sequence([0.55] * 2000) > lower_sequence([0.55] * 1000)


def test_promotion_spends_error_budget_across_candidate_attempts():
    rows = [{"pair": i, "scores": [float(i % 100 < 56)] * 2} for i in range(4096)]
    first = promotion_evidence(rows, 1)
    later = promotion_evidence(rows, 10)
    assert first["passed"]
    assert first["lower"] > 0.5
    assert first["alpha"] == 0.025
    assert later["threshold"] > first["threshold"]
    assert later["lower"] < first["lower"]
    with pytest.raises(ValueError, match="duplicate"):
        promotion_evidence(rows + rows[:1], 1)


def test_null_mixture_is_a_supermartingale_for_all_possible_pair_results():
    # Exact enumeration of the first three binary observations at the boundary.
    # E[E_t] == 1; stopping early at a fixed threshold cannot inflate it.
    from itertools import product

    wealth = [np.exp(log_evidence(xs, 0.5)) for xs in product((0.0, 1.0), repeat=3)]
    assert np.mean(wealth) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        log_evidence([float("nan")], 0.5)
