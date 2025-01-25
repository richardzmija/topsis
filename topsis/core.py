import numpy as np
from typing import Optional, Tuple


class Topsis:
    """
    Implements TOPSIS multi-criteria decision analysis method.
    """

    def __init__(
        self,
        n_alternatives: int,
        n_criteria: int,
        weights: np.ndarray,
        decision_matrices: np.ndarray,
        is_benefit_criteria: Optional[np.ndarray] = None,
    ):
        """
        Initialize the TOPSIS object.

        Args:
            n_alternatives (int): The number of alternatives (encoded as integers 0...N-1).
            n_criteria (int): The number of criteria (encoded as natural numbers 0...C-1).
            weights (ndarray): A 1D array of length n_criteria representing the weights of each criterion.
            decision_matrices (ndarray): A 3D tensor of shape (n_experts, n_alternatives, n_criteria)
                representing the decision matrices from multiple experts.
            is_benefit_criteria: (Optional[ndarray]): A boolean array of shape (n_criteria,) indicating whether each
                criterion is a "benefit" criterion (True) or a "cost" criterion (False). If None, all criteria are
                assumed to be benefit criteria. Defaults to None.
        Returns:
            None
        """
        self.n_alternatives = n_alternatives
        self.n_criteria = n_criteria

        if np.any(weights <= 0):
            raise ValueError("Error: All weight values must be positive.")

        self.weights = weights / weights.sum()  # Normalize the weight vector.
        self.decision_matrices = decision_matrices

        if is_benefit_criteria is None:
            self.is_benefit_criteria = np.ones(n_criteria, dtype=bool)
        else:
            self.is_benefit_criteria = is_benefit_criteria

        if len(self.weights) != self.n_criteria:
            raise ValueError("The length of weights must match n_criteria.")
        if (
            self.decision_matrices.shape[1] != self.n_alternatives
            or self.decision_matrices.shape[2] != self.n_criteria
        ):
            raise ValueError(
                "decision_matrices must have shape (n_experts, n_alternatives, n_criteria)."
            )
        if len(self.is_benefit_criteria) != self.n_criteria:
            raise ValueError("The length of is_benefit_criteria must match n_criteria.")

    def _aggregate_expert_opinions(self) -> np.ndarray:
        """
        Aggregate the decision matrices from multiple experts into a single decision matrix.

        Returns:
            ndarray: A 2D array of shape (n_alternatives, n_criteria) representing
                the aggregated decision matrix.
        """
        return np.mean(self.decision_matrices, axis=0)

    def _normalize_decision_matrix(self, decision_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the decision matrix using the Euclidean norm.

        Args:
            decision_matrix (ndarray): A 2D array of shape (n_alternatives, n_criteria).

        Returns:
            ndarray: A 2D array of shape (n_alternatives, n_criteria) with each column
                normalized by its Euclidean norm.
        """
        # Compute the norm for each criterion (column).
        norm = np.sqrt(np.sum(decision_matrix**2, axis=0))

        # To avoid division by zero replace it with multiplicative identity.
        norm[norm == 0] = 1.0

        return decision_matrix / norm

    def _apply_weights(self, normalized_matrix: np.ndarray) -> np.ndarray:
        """
        Apply criteria weights to the normalized decision matrix.

        Args:
            normalized_matrix (ndarray): A 2D array of shape (n_alternatives, n_criteria) after normalization.

        Returns:
            ndarray: A 2D array of shape (n_alternatives, n_criteria) after multiplying each column by the
                corresponding criterion weight.
        """
        # Perform element-wise multiplication after broadcasting.
        return normalized_matrix * self.weights

    def _determine_ideal_solutions(
        self, weighted_matrix: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Determine the ideal and negative-ideal solution vectors.

        Args:
            weighted_matrix (ndarray): A 2D array of shape (n_alternatives, n_criteria) after weighting.

        Returns:
            Tuple[ndarray, ndarray]: The first element is a 1D array representing the ideal solution.
                The second element is a 1D array representing the negative-ideal solution.
        """
        ideal = np.zeros(self.n_criteria)
        negative_ideal = np.zeros(self.n_criteria)

        for j in range(self.n_criteria):
            column = weighted_matrix[:, j]

            if self.is_benefit_criteria[j]:
                ideal[j] = np.max(column)
                negative_ideal[j] = np.min(column)
            else:
                ideal[j] = np.min(column)
                negative_ideal[j] = np.max(column)

        return ideal, negative_ideal

    def _calculate_distances_to_ideals(
        self, weighted_matrix: np.ndarray, ideal: np.ndarray, negative_ideal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Euclidean distance of each alternative from the ideal and
        negative-ideal solutions.

        Args:
            weighted_matrix (ndarray): A 2D array of shape (n_alternatives, n_criteria).
            ideal (ndarray): A 1D array of length n_criteria (best solution).
            negative_ideal (ndarray): A 1D array of length n_criteria (worst solution).

        Returns:
            Tuple[ndarray, ndarray]: The first element is a 1D array of length n_alternatives containing
            the distance to the ideal solution. The second element is a 1D array of length n_alternatives
            containing distance to the negative-ideal solution.
        """
        dist_to_ideal = np.sqrt(np.sum((weighted_matrix - ideal) ** 2, axis=1))
        dist_to_negative_ideal = np.sqrt(
            np.sum((weighted_matrix - negative_ideal) ** 2, axis=1)
        )
        return dist_to_ideal, dist_to_negative_ideal

    def _calculate_closeness_scores(
        self, dist_to_ideal: np.ndarray, dist_to_negative_ideal: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the relative closeness of each alternative to the ideal solution.
        A higher closeness score for an alternative indicates a closer proximity to
        the ideal solution and hence a better alternative.

        Args:
            dist_to_ideal (ndarray): A 1D array of shape (n_alternatives,) containing the distance
                to the ideal solution.
            dist_to_negative_ideal (ndarray): A 1D array of shape (n_alternatives,) containing the
                distance to the negative-ideal solution.

        Returns:
            ndarray: A 1D array of shape (n_alternatives,) containing the closeness score for each alternative.
        """
        if np.any((dist_to_ideal == 0) & (dist_to_negative_ideal == 0)):
            raise ValueError(
                "Error: Zero denominator detected in relative closeness calculation for one or more alternatives."
            )

        return dist_to_negative_ideal / (dist_to_ideal + dist_to_negative_ideal)

    def rank_alternatives(self) -> np.ndarray:
        """
        Calculate the ranking for the alternatives.

        Returns:
            ndarray: A 1D array of shape (n_alternatives,) containing the ranking of alternatives.
            Note: The array contains the indices of alternatives (0-based).
        """
        aggregated_matrix = self._aggregate_expert_opinions()
        normalized_matrix = self._normalize_decision_matrix(aggregated_matrix)
        weighted_matrix = self._apply_weights(normalized_matrix)
        ideal, negative_ideal = self._determine_ideal_solutions(weighted_matrix)
        dist_to_ideal, dist_to_negative_ideal = self._calculate_distances_to_ideals(
            weighted_matrix, ideal, negative_ideal
        )
        scores = self._calculate_closeness_scores(dist_to_ideal, dist_to_negative_ideal)
        ranking = np.argsort(scores)[::-1].copy()
        return ranking
