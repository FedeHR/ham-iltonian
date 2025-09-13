import numpy as np
import pennylane as qml
import copy

from hamiltonians.problems.instance_generators import generate_n_maxcut_instances, create_maxcut_instance
from hamiltonians.algorithms.meta_map import MetaPolyRBF
from hamiltonians.algorithms.trainer import es_train_driver
from hamiltonians.algorithms.vqa import VQA

def main():
    # --- Configuration ---
    N = 8
    p = 0.4
    num_train = 60
    seed = 1234

    # Meta model params
    deg = 2
    R = 5
    
    # Meta-Training (ES) params
    iterations = 50 # Meta-iterations
    batch = 12
    es_eps = 0.08
    es_eta = 0.35
    a_meta = 0.08
    c_meta = 0.06
    
    # Lambda range for training
    lam_min, lam_max = -1.0, 1.0

    np.random.seed(seed)
    qml.numpy.random.seed(seed)

    # 1. Build training instances using the new generator
    print("--- Generating Training Instances ---")
    train_instances = generate_n_maxcut_instances(n_instances=num_train, n_nodes=N,
                                                  n_edge_params=2, edge_probability=p, base_seed=seed)

    # 2. QAOA machinery setup
    p_layers = 10
    out_dim = 2 * p_layers
    
    # 3. Meta model
    meta_qaoa = MetaPolyRBF(out_dim=out_dim, polynomial_degree=deg, n_rbf_kernels=R, lambda_min=lam_min, lambda_max=lam_max, seed=seed)

    # 4. Objective function
    def objective(params_vec, problem_instance, lam):
        vqa = VQA(problem_instance, ansatz="qaoa", layers=p_layers)
        loss, shots = vqa.parametrized_expval_loss(params_vec, modifier_function="quadratic_edge_scaling", scaling_factor=lam)
        return loss, shots


    # 5. Meta-Training
    print("\n--- Starting Meta-Training for QAOA ---")
    meta_qaoa = es_train_driver(meta_qaoa, train_instances, objective, lam_min=lam_min, lam_max=lam_max,
                                meta_iterations=iterations, inner_iterations=1000, batch_size=batch, es_eps=es_eps,
                                es_eta=es_eta, a_spsa=a_meta, c_spsa=c_meta, seed=seed)
    print("--- Meta-Training Finished ---")

    # 6. Evaluation
    print("\n--- Evaluating Trained Meta-Model ---")
    test_instance_base = create_maxcut_instance(n_nodes=N, edge_probability=p, n_edge_params=2, seed=seed + 1000)
    
    for lam_test in [-0.75, 0.0, 0.75]:
        # Get parameters from meta-model
        params_from_meta_flat = meta_qaoa.g(lam_test)
        
        # Reshape parameters for the QAOA ansatz
        params_from_meta = params_from_meta_flat.reshape(p_layers, 2)

        # Calculate energy using the same objective function
        loss, shots = objective(params_from_meta, test_instance_base, lam_test)
        
        # For comparison, solve the lambda-modified problem classically
        test_problem_modified = copy.deepcopy(test_instance_base)
        test_problem_modified.modify_parameters("quadratic_edge_scaling", scaling_factor=lam_test)
        
        classical_solution = test_problem_modified.solve_classically()
        classical_energy = classical_solution['cut_value']
        
        print(f"\nLambda = {lam_test:.2f}")
        print(f"  Energy from Meta-QAOA: {loss:.4f}")
        print(f"  Classical MaxCut value: {classical_energy:.4f}")
        # Approximation ratio can be misleading if classical_energy is 0 or negative
        if classical_energy > 1e-6:
            print(f"  Approximation Ratio: {loss / classical_energy:.4f}")

if __name__ == "__main__":
    main()
