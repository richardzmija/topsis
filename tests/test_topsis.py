"""
The most essential tests to verify that the algorithm
works correctly for the most common types of inputs.
"""

import numpy as np

from topsis import Topsis


def test_single_expert_benefit_criteria():
    """
    Test with 1 expert, 3 alternatives, 2 criteria.
    """
    decision_matrices = np.array([[[3.0, 4.0], [2.0, 2.0], [5.0, 6.0]]])

    n_alternatives = 3
    n_criteria = 2
    weights = np.array([0.5, 0.5])

    topsis = Topsis(
        n_alternatives=n_alternatives,
        n_criteria=n_criteria,
        weights=weights,
        decision_matrices=decision_matrices,
    )

    ranking = topsis.rank_alternatives()

    assert ranking[0] == 2, "Expected alternative #2 to be first."
    assert ranking[1] == 0, "Expected alternative #0 to be second."
    assert ranking[2] == 1, "Expected alternative #1 to be third."


def test_multiple_experts_aggregation():
    """
    Test aggregation when multiple experts provide data.
    """
    decision_matrices = np.array([[[1.0, 2.0], [3.0, 4.0]], [[2.0, 4.0], [2.0, 2.0]]])

    n_alternatives = 2
    n_criteria = 2
    weights = np.array([0.4, 0.6])

    topsis = Topsis(
        n_alternatives=n_alternatives,
        n_criteria=n_criteria,
        weights=weights,
        decision_matrices=decision_matrices,
    )

    expected_aggregated = np.mean(decision_matrices, axis=0)
    actual_aggregated = topsis._aggregate_expert_opinions()

    np.testing.assert_array_almost_equal(
        actual_aggregated,
        expected_aggregated,
        err_msg="Aggregated matrix differs from expected mean.",
    )

    ranking = topsis.rank_alternatives()
    # Make sure there is no error.
    assert len(ranking) == 2
