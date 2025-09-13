from __future__ import annotations
from typing import Callable, List
import numpy as np
from hamiltonians.algorithms.meta_map import MetaPolyRBF
from hamiltonians.problems.maxcut import MaxCutProblem

def es_train_driver(model: MetaPolyRBF, training_instances: List[MaxCutProblem],
                    objective_from_params: Callable[[np.ndarray, MaxCutProblem, float], tuple[float, int]],
                    lam_min: float, lam_max: float, meta_iterations: int, inner_iterations: int, batch_size: int,
                    es_eps: float, es_eta: float, a_spsa, c_spsa, seed: int) -> MetaPolyRBF:

    rng = np.random.default_rng(seed)
    phi = model.phi.copy()
    
    for i in range(meta_iterations):
        print(f"Meta-iteration {i+1}/{meta_iterations}")

        lambda_values = rng.uniform(lam_min, lam_max, size=batch_size)
        instances = rng.choice(training_instances, size=batch_size)
        tasks = list(zip(instances, lambda_values))

        delta = rng.choice([-1.0, +1.0], size=phi.shape)

        def meta_obj(phi_vec: np.ndarray) -> float:
            model.phi = phi_vec
            vals = []
            p_layers = model.out_dim // 2

            for inst, lam in tasks:
                theta0_flat = model.g(float(lam))
                theta0 = theta0_flat.reshape(p_layers, 2)
                theta = theta0.copy()

                if inner_iterations > 0:
                    rng_loc = np.random.default_rng(rng.integers(1_000_000_000))
                    delta_th = rng_loc.choice([-1.0, +1.0], size=theta.shape)
                    E_plus, _ = objective_from_params(theta + c_spsa * delta_th, inst, float(lam))
                    E_minus, _ = objective_from_params(theta - c_spsa * delta_th, inst, float(lam))
                    ghat = (E_plus - E_minus) / (2.0 * c_spsa) * delta_th
                    theta = theta + a_spsa * ghat

                loss, shots = objective_from_params(theta, inst, float(lam))
                vals.append(loss)
                return float(np.mean(vals))

        Jp = meta_obj(phi + es_eps * delta)
        Jm = meta_obj(phi - es_eps * delta)
        grad = (Jp - Jm) / (2.0 * es_eps) * delta
        phi = phi + es_eta * grad
        model.phi = phi
    return model
