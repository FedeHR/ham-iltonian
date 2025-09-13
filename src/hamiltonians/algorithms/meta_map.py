from __future__ import annotations
import numpy as np


class MetaPolyRBF:
    def __init__(self, out_dim: int, polynomial_degree: int = 2, n_rbf_kernels: int = 5,
                 lambda_min: float = -1.0, lambda_max: float = -1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.out_dim = out_dim
        self.polynomial_degree = polynomial_degree
        self.n_rbf_kernels = n_rbf_kernels
        self.rbf_centers = np.linspace(lambda_min, lambda_max, n_rbf_kernels)
        
        median_center_distance = np.median(np.diff(self.rbf_centers)) if n_rbf_kernels > 1 else 1.0
        self.rbf_width = max(1e-6, median_center_distance)
        
        self.polynomial_coeffs = 0.1 * rng.standard_normal((out_dim, polynomial_degree + 1))
        self.rbf_coeffs = 0.1 * rng.standard_normal((out_dim, n_rbf_kernels))

    @property
    def phi(self) -> np.ndarray:
        return np.concatenate([self.polynomial_coeffs.ravel(), self.rbf_coeffs.ravel()])

    @phi.setter
    def phi(self, vec: np.ndarray):
        n_poly_features = self.polynomial_degree + 1
        n_poly_coeffs = self.out_dim * n_poly_features
        self.polynomial_coeffs = vec[:n_poly_coeffs].reshape(self.out_dim, n_poly_features)
        self.rbf_coeffs = vec[n_poly_coeffs:].reshape(self.out_dim, self.n_rbf_kernels)

    def g(self, lambda_val: float) -> np.ndarray:
        poly_features = np.array([lambda_val ** k for k in range(self.polynomial_degree + 1)])
        rbf_features = np.exp(-0.5 * ((lambda_val - self.rbf_centers) ** 2) / (self.rbf_width ** 2))
        return self.polynomial_coeffs @ poly_features + self.rbf_coeffs @ rbf_features
