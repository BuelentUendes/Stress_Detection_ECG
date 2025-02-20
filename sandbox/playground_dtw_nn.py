# Sample script to learn how to do DTW-KNN classifier for time-series data which might be interesting in our case


# Sample script to investigate a fast version of DTW algorithm which i can plug in in ContraLSP

import numpy as np
import torch
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from time import time
from tslearn.metrics import dtw, soft_dtw
from tslearn.barycenters import dtw_barycenter_averaging
from scipy.optimize import minimize
from cardano_method import CubicEquation


def generate_synthetic_timeseries(
        n_samples: int,
        seq_length: int,
        feature_dim: int,
        noise_level: float = 0.1,
        random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate synthetic time series data.

    Args:
        n_samples: Number of time series to generate
        seq_length: Length of each time series
        feature_dim: Dimension of features at each time step
        noise_level: Standard deviation of Gaussian noise
        random_seed: Random seed for reproducibility

    Returns:
        Array of shape (n_samples, seq_length, feature_dim)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate base signals using sine waves with different frequencies
    t = np.linspace(0, 4 * np.pi, seq_length)
    series = []

    for _ in range(n_samples):
        # Generate random frequencies for each feature dimension
        freqs = np.random.uniform(0.5, 2.0, size=feature_dim)
        amplitudes = np.random.uniform(0.5, 2.0, size=feature_dim)

        # Generate the signal
        signal = np.zeros((seq_length, feature_dim))
        for i in range(feature_dim):
            signal[:, i] = amplitudes[i] * np.sin(freqs[i] * t)

        # Add noise
        noise = np.random.normal(0, noise_level, size=(seq_length, feature_dim))
        signal += noise
        series.append(signal)

    return np.array(series)


def get_similarity_time_series(ts_1, ts_2, metric="softdtw"):
    """
    Get the similarity between two times-series
    :param ts_1:
    :param ts_2:
    :return:
    """

    if metric == "dtw":

        ts_1_zeroes = np.zeros_like(ts_1)
        ts_2_zeroes = np.zeros_like(ts_2)

        similarity = dtw(ts_1, ts_1_zeroes) + dtw(ts_2, ts_2_zeroes) + dtw(ts_1, ts_2)
        similarity /= 2

    elif metric == "softdtw":

        soft_dtw_distance_x_y = soft_dtw(ts_1, ts_2)
        soft_dtw_distance_x_x = soft_dtw(ts_1, ts_1)
        soft_dtw_distance_y_y = soft_dtw(ts_2, ts_2)
        similarity = soft_dtw_distance_x_y - 0.5 * (soft_dtw_distance_x_x + soft_dtw_distance_y_y)
        print(similarity)

    else:
        raise NotImplementedError(f"{metric} is not implemented. Choices are: 'dtw' and 'softdtw'.")

    return similarity


def get_partial_observed_similarity_matrix(time_series: np.ndarray, metric="softdtw") -> np.ndarray:
    """
    Generate the partially observed time-series
    Approach is based on:
    https://arxiv.org/abs/1702.03584
    :param time_series: n x d array
    :return: n x n similarity matrix, where 0 indicates unobserved similarity
    """

    n_samples = time_series.shape[0]
    if metric == "softdtw":
        # Softdtw is bounded from below 0 so -1 is a value which can indiate missingness
        similarity_matrix = np.ones((n_samples, n_samples)) * -1.

    else:
        similarity_matrix = np.zeros((n_samples, n_samples))

    # We sample n log n samples
    number_of_observed_instances = int(n_samples * np.log(n_samples))
    # number_of_observed_instances = int(n_samples * n_samples)

    possible_indices = np.arange(0, n_samples * n_samples)
    chosen_random_indices = np.random.choice(possible_indices, number_of_observed_instances, replace=False)

    for random_index in chosen_random_indices:
        ts_1_idx = int(random_index // n_samples)
        ts_2_idx = int(random_index % n_samples)

        ts_similarity = get_similarity_time_series(time_series[ts_1_idx], time_series[ts_2_idx])
        similarity_matrix[ts_1_idx, ts_2_idx] = ts_similarity

    # Check that we have exactly number_of_observed instances non-zero
    # (we filled in x,y and y,x as the matrix is symmetric)
    # assert int(np.count_nonzero(similarity_matrix)) == int(number_of_observed_instances)
    if metric == "dtw":
        assert int(np.count_nonzero(similarity_matrix)) == int(number_of_observed_instances)
    else:
        # We have a very sparse matrix of size n_samples x n_samples so we can check filled values
        assert np.sum(similarity_matrix == -1.) == (n_samples * n_samples - number_of_observed_instances)

    return similarity_matrix


def get_full_similarity_matrix(time_series):
    """
       Generate the fully-observed time-series
       Approach is based on:
       https://arxiv.org/abs/1702.03584
       :param time_series: n x d array
       :return: n x n similarity matrix, where 0 indicates unobserved similarity
       """

    n_samples = time_series.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            ts_similarity = get_similarity_time_series(time_series[i], time_series[j])
            similarity_matrix[i, j] = ts_similarity

    return similarity_matrix


def get_omega_matrix(similarity_matrix):
    if torch.is_tensor(similarity_matrix):
        omega_matrix = (similarity_matrix > 0.).float()

    else:
        omega_matrix = (similarity_matrix > 0.).astype(float)
    return omega_matrix


def cubic_root(d):
    if d < 0.0:
        return -cubic_root(-d)
    else:
        return d ** (1.0 / 3.0)


def root_c(a, b):
    a3 = 4 * (a ** 3)
    b2 = 27 * (b ** 2)
    delta = a3 + b2

    if delta <= 0:  # 3 distinct real roots or 1 real multiple solution
        r3 = 2 * np.sqrt(-a / 3)
        th3 = np.arctan2(np.sqrt(-delta / 108), -b / 2) / 3
        ymax = 0
        xopt = 0
        for k in range(0, 5, 2):
            x = r3 * np.cos(th3 + ((k * 3.14159265) / 3))
            y = (x ** 4) / 4 + a * (x ** 2) / 2 + b * x
            if y < ymax:
                ymax = y
                xopt = x
        return xopt
    else:  # 1 real root and two complex
        z = np.sqrt(delta / 27)
        x = cubic_root(0.5 * (-b + z)) + cubic_root(0.5 * (-b - z))
        return x


def cyclic_coordinate_descent_algorithm(similarity_matrix, d=3, number_iterations=10):
    """
    Implementation of the coordinate descent algorithm as proposed in:
        https://arxiv.org/abs/1702.03584
    :param similarity_matrix:
    :param d:
    :param number_iterations:
    :return:
    """

    # Initialization:
    n = similarity_matrix.shape[0]
    omega_matrix = get_omega_matrix(similarity_matrix)

    # X = np.zeros((n, d))
    X = np.random.rand(n, d) * 1e-4

    # Apply projection P_Omega
    R = omega_matrix * (similarity_matrix - X @ X.T)

    for iter in range(number_iterations):
        X_new = X.copy()
        for feature_dim in range(d):
            R += omega_matrix * np.outer(X_new[:, feature_dim], X_new[:, feature_dim])

            for number_sample in range(n):
                observed_indices = np.where(omega_matrix[number_sample, :] > 0)[0]
                # observed_indices = np.where(omega_matrix[:, feature_dim] > 0)[0]
                p, q = 0.0, 0.0

                for k in observed_indices:
                    p += X_new[k, feature_dim] ** 2
                    q -= X_new[k, feature_dim] * R[number_sample, k]

                p -= X_new[number_sample, feature_dim] ** 2 + R[number_sample, number_sample]
                q += X_new[number_sample, feature_dim] * R[number_sample, number_sample]

                X_new[number_sample, feature_dim] = root_c(p, q)

            R -= omega_matrix * np.outer(X_new[:, feature_dim], X_new[:, feature_dim])

        X = X_new.copy()

    return X


def torch_cyclic_coordinate_descent_algorithm(similarity_matrix: torch.Tensor, d=3,
                                              number_iterations=10) -> torch.Tensor:
    """
    Implementation of the coordinate descent algorithm as proposed in:
        https://arxiv.org/abs/1702.03584

    Args:
        similarity_matrix: Similarity matrix (n_samples x n_samples)
        d: Latent dimensionality
        number_iterations: Number of iterations for coordinate descent

    Returns:
        Feature matrix X (n_samples x d)
    """
    # Initialization
    n = similarity_matrix.size(0)
    omega_matrix = get_omega_matrix(similarity_matrix)
    X = torch.rand((n, d)) * 1e-4

    # Apply projection P_Omega
    R = omega_matrix * (similarity_matrix - X @ X.T)

    for iter in range(number_iterations):
        X_new = X.clone()
        for feature_dim in range(d):
            R += omega_matrix * torch.outer(X_new[:, feature_dim], X_new[:, feature_dim])

            for number_sample in range(n):
                observed_indices = torch.where(omega_matrix[number_sample, :] > 0)[0]
                p, q = 0.0, 0.0

                for k in observed_indices:
                    p += X_new[k, feature_dim] ** 2
                    q -= X_new[k, feature_dim] * R[number_sample, k]

                p -= X_new[number_sample, feature_dim] ** 2 + R[number_sample, number_sample]
                q += X_new[number_sample, feature_dim] * R[number_sample, number_sample]

                p_np = p.cpu().numpy()
                q_np = q.cpu().numpy()
                cubic_root = root_c(p_np, q_np)
                print(f"The root is {cubic_root}")
                X_new[number_sample, feature_dim] = torch.tensor(cubic_root)

            R -= omega_matrix * torch.outer(X_new[:, feature_dim], X_new[:, feature_dim])

        X = X_new.clone()

    return X


# Example usage
if __name__ == "__main__":
    # Generate sample time series
    n_samples = 50  # Increased to better demonstrate barycenter
    seq_length = 200
    feature_dim = 3

    series = generate_synthetic_timeseries(
        n_samples=n_samples,
        seq_length=seq_length,
        feature_dim=feature_dim,
        random_seed=42
    )

    similarity_matrix = get_partial_observed_similarity_matrix(series)
    similarity_matrix_torch = torch.from_numpy(similarity_matrix)
    full_similarity_matrix = get_full_similarity_matrix(series)

    X = cyclic_coordinate_descent_algorithm(similarity_matrix, d=3, number_iterations=25)
    X_torch = torch_cyclic_coordinate_descent_algorithm(similarity_matrix_torch, d=3, number_iterations=25)
    learned_similarity = X @ X.T
    learned_similarity_torch = X_torch @ X_torch.T

    difference = full_similarity_matrix - learned_similarity

    error = difference / np.linalg.norm(difference)

    # Calculate DTW distance between first two series using tslearn
    start_time = time()
    distance = dtw(series[0], series[1])
    end_time = time()

    similarity = get_similarity_time_series(series[0], series[1])
    similarity2 = get_similarity_time_series(series[1], series[0])
    similarity3 = get_similarity_time_series(series[1], series[1])
    print(f"Similarity 1 is {similarity}")
    print(f"Similarity 2 is {similarity2}")
    print(f"Similarity 3 is {similarity3}")

    print(f"DTW distance between series: {distance:.4f}")
    print(f"Computation time: {(end_time - start_time) * 1000:.2f} ms")

    # Calculate barycenter
    start_time = time()
    barycenter = dtw_barycenter_averaging(series)
    end_time = time()

    print(f"Barycenter computation time: {(end_time - start_time) * 1000:.2f} ms")

    # Plot the first dimension of all time series and the barycenter
    plt.figure(figsize=(12, 6))

    # Plot original series
    for i in range(n_samples):
        plt.plot(series[i][:, 0], alpha=0.5, label=f'Series {i + 1}')

    # Plot barycenter
    plt.plot(barycenter[:, 0], 'k-', linewidth=2, label='Barycenter')

    plt.legend()
    plt.title('First dimension of generated time series and their barycenter')
    plt.show()

