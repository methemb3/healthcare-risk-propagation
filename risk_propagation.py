"""
Healthcare Risk Propagation Algorithm
-------------------------------------

This program demonstrates a prototype implementation of iterative
risk propagation across a patient network using eigenvector methods.

The algorithm models patients as nodes in a graph where edges represent
shared health risk factors such as:

    - Shared chronic diseases
    - Genetic similarity
    - Geographic proximity
    - Similar environmental exposure
    - Shared healthcare providers

Risk propagates across the network until the system reaches equilibrium.

Author: Methembe Moses Ncube
Purpose: Risk propagation prototype inspired by Mario Schlosser's
         healthcare risk equilibrium theorem for healthcare networks.
"""

import numpy as np


class HealthcareRiskNetwork:
    """
    Graph-based healthcare risk prediction system.

    The system stores:
        A : adjacency matrix (patient relationships)
        b : baseline patient risk scores
        r : propagated risk scores
    """

    def __init__(self, adjacency_matrix, baseline_risk, alpha=0.85):
        """
        Initialize the healthcare network.

        Parameters
        ----------
        adjacency_matrix : numpy array
            NxN matrix representing relationships between patients.

        baseline_risk : numpy array
            Vector of baseline clinical risks derived from features
            such as age, BMI, blood pressure, etc.

        alpha : float
            Risk propagation coefficient (0 <= alpha <= 1).

            alpha near 1 → network effects dominate
            alpha near 0 → baseline risk dominates
        """

        self.A = adjacency_matrix
        self.b = baseline_risk
        self.alpha = alpha

        # Initialize propagated risk with baseline values
        self.r = baseline_risk.copy()

    def normalize_graph(self):
        """
        Normalize adjacency matrix so rows sum to 1.

        This ensures the propagation behaves like a probability
        distribution and prevents risk explosion during iteration.
        """

        row_sums = self.A.sum(axis=1)

        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                self.A[i] = self.A[i] / row_sums[i]

    def propagate_risk(self, max_iterations=100, tolerance=1e-6):
        """
        Perform iterative risk propagation.

        The update rule is:

            r(t+1) = alpha * A * r(t) + (1 - alpha) * b

        Iteration continues until convergence.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations.

        tolerance : float
            Convergence threshold.
        """

        for iteration in range(max_iterations):

            new_r = (
                self.alpha * np.dot(self.A, self.r)
                + (1 - self.alpha) * self.b
            )

            # Check convergence
            diff = np.linalg.norm(new_r - self.r)

            if diff < tolerance:
                print(f"Converged after {iteration} iterations.")
                break

            self.r = new_r

        return self.r

    def top_risk_patients(self, k=5):
        """
        Identify the highest-risk patients after propagation.

        Returns the indices of patients with the largest risk scores.
        """

        return np.argsort(self.r)[::-1][:k]


# ---------------------------------------------------------
# Example Simulation
# ---------------------------------------------------------

if __name__ == "__main__":

    """
    Example network with 6 patients.

    Edges represent shared risk relationships such as:
        - same neighborhood
        - similar health conditions
        - shared physician
    """

    adjacency_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ], dtype=float)

    """
    Baseline risk scores derived from clinical variables.

    Example:
        patient 0 → young, low BMI → low risk
        patient 3 → hypertension + obesity → higher risk
    """

    baseline_risk = np.array([
        0.1,
        0.2,
        0.2,
        0.6,
        0.3,
        0.4
    ])

    # Create network model
    network = HealthcareRiskNetwork(adjacency_matrix, baseline_risk)

    # Normalize graph
    network.normalize_graph()

    # Run propagation
    final_risk_scores = network.propagate_risk()

    print("\nFinal Risk Scores:")
    print(final_risk_scores)

    print("\nHighest Risk Patients:")
    print(network.top_risk_patients())
