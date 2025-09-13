
import random
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import operator

from hamiltonians.problems.base import Problem
from .circuits.pennylane_circuits import *

class VQA:
    def __init__(self, problem: Problem, ansatz: str, layers: int, seed=0, gpu=False, init_params=None):
        
        # Seeding
        random.seed(seed)
        np.random.seed(seed)
        self.seed = seed

        # Problem
        self.problem = problem

        # Hamiltonian
        if self.problem.hamiltonian is None:
            self.problem.build_hamiltonian()
        self.cost_hamiltonian = self.problem.get_hamiltonian().to_pennylane()  # Mixer defined on pennylane_circuits.py
        self.n_qubits = self.problem.get_n_qubits()

        # Device
        if gpu:
            self.dev = qml.device("lightning.gpu", wires=self.n_qubits, seed=self.seed)
        else:
            self.dev = qml.device("default.qubit", wires=self.n_qubits, seed=self.seed)

        # Ansatz
        self.ansatz = ansatz
        self.n_layers = layers
        self._init_circuit_including_measurement()

        # Initial circuit parameters
        if init_params is None:
            self.init_params = init_params_from_ansatz(self, ansatz)
        else:
            self.init_params = init_params
        self.n_params = self.init_params.size
        
        # Optimized parameters
        self.current_params = self.init_params.copy()
        
        # Optimization parameters
        self.optimizer = None
        self.iterations = 0
        self.loss_func = None

        # CVaR / Ascending CVaR parameters
        self.init_alpha = 0
        self.ascending_rate = 0
        self.ascending_steps = 0
        self.init_shots = 0  # Shots are adaptively chosen depending on alpha as int(init_shots/alpha)

        # Logs
        self.param_history = []
        self.loss_history = []
        self.approximation_ratios = []


    def _init_circuit_including_measurement(self):
        """
        Initialize the full circuit, including the Ansatz and the "measurement" operations (expval, sample, probs and count).
        """

        if self.ansatz == "qaoa_new":
            self.max_bound = 10  # After experiments sent to Jonas, this appears to be the best value. Explain in BA

            self.expval = qml.QNode(lambda *args: qaoa_new_expval(*args, max_cut_bound=self.max_bound), device=self.dev)  # Lambda function to partially apply the max_bound argument
            self.sample = qml.QNode(lambda *args: qaoa_new_sample(*args, max_cut_bound=self.max_bound), device=self.dev)
            self.probs = qml.QNode(lambda *args: qaoa_new_probs(*args, max_cut_bound=self.max_bound), device=self.dev)
            self.counts = qml.QNode(lambda *args: qaoa_new_counts(*args, max_cut_bound=self.max_bound), device=self.dev)

        elif self.ansatz == "qaoa":
            self.expval = qml.QNode(qaoa_expval, device=self.dev)
            self.sample = qml.QNode(qaoa_sample, device=self.dev)
            self.probs = qml.QNode(qaoa_probs, device=self.dev)
            self.counts = qml.QNode(qaoa_counts, device=self.dev)

        elif self.ansatz == "vqe_barkoutsos":
            self.expval = qml.QNode(barkoutsos_expval, device=self.dev)
            self.sample = qml.QNode(barkoutsos_sample, device=self.dev)
            self.probs = qml.QNode(barkoutsos_probs, device=self.dev)
            self.counts = qml.QNode(barkoutsos_counts, device=self.dev)
            # self.measure = qml.QNode(barkoutsos_measure, device=self.dev)

        elif self.ansatz == "vqe_efficient_su2":
            self.expval = qml.QNode(expval_efficient_su2, device=self.dev)
            self.sample = qml.QNode(sample_efficient_su2, device=self.dev)
            self.probs = qml.QNode(probs_efficient_su2, device=self.dev)
            self.counts = qml.QNode(counts_efficient_su2, device=self.dev)

        elif self.ansatz == "vqe_new":
            self.expval = qml.QNode(new_vqe_expval, device=self.dev)
            self.sample = qml.QNode(new_vqe_sample, device=self.dev)
            self.probs = qml.QNode(new_vqe_probs, device=self.dev)
            self.counts = qml.QNode(new_vqe_counts, device=self.dev)

    def parametrized_expval_loss(self, circuit_params, modifier_function="quadratic_edge_scaling", **kwargs):
        """
        Apply a given modifier function to the problem configuration and evaluate the expectation value loss afterwards.
        circuit_params: np.array
            Parameters of the circuit (e.g. QAOA)
        modifier_function: str
            Name of the modifier function (e.g. "quadratic_edge_scaling") to be applied on the problem instance.
        **kwargs:
            Keyword arguments passed to the modifier function, i.e. scaling_factor (lambda).
        """
        self.problem.modify_parameters(modifier_function, **kwargs)
        loss = self.expval_loss(circuit_params)
        # The '0' for calls is a placeholder as we aren't tracking shots for now
        return loss, 0

    def expval_loss(self, params):
        loss = self.problem.hamiltonian.constant - self.expval(params, self.n_qubits, self.n_layers, self.cost_hamiltonian)
        return loss

    #
    # # TODO CVaR / Ascending CVaR loss (already implemented in vqa.py)

    def optimize(self, max_iter=None, optimizer="ADAM"):
        if max_iter is None:
            max_iter = 66 * self.n_qubits

        if optimizer == "ADAM":
            self.optimizer = qml.AdamOptimizer()
        elif optimizer == "SPSA":
            self.optimizer = qml.SPSAOptimizer()

        for i in range(max_iter):
            self.iterations += 1
            self.current_params, loss_value = self.optimizer.step_and_cost(
                self.expval_loss, self.current_params)
            self.loss_history.append(loss_value)
            self.param_history.append(self.current_params)

    def draw_circuit(self):
        (qml.draw_mpl(self.sample, style='pennylane')
         (self.current_params, self.n_qubits, self.n_layers, self.cost_hamiltonian))

    def plot_optimization_history(self):
        """
        Plots the loss history during optimization.
        """
        if not self.loss_history:
            print("No optimization history to plot. Run the optimize method first.")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Loss')
        plt.title('Optimization History')
        plt.xlabel('Iteration')
        plt.ylabel('Loss (Expectation Value)')
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_top_solutions(self, top_n=10, shots=1000):
        counts = self.counts(self.current_params, self.n_qubits, self.n_layers, self.cost_hamiltonian, shots=shots)
        sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
        
        top_solutions = []
        print(f"Top {top_n} solution candidates from VQA:")
        for i in range(min(top_n, len(sorted_counts))):
            bitstring, count = sorted_counts[i]
            probability = count / shots
            top_solutions.append((bitstring, probability))
            print(f"  {i+1}. Bitstring: {bitstring}, Probability: {probability:.4f}")
        
        return top_solutions

    def visualize_vqa_solution(self, bitstring=None):
        if bitstring is None:
            # Get the most probable solution if no bitstring is provided
            top_solutions = self.get_top_solutions(top_n=1)
            if not top_solutions:
                print("Could not find any solutions to visualize.")
                return
            bitstring = top_solutions[0][0]
            print(f"Found top solution with bitstring: {bitstring}")
        else:
            print(f"--- Visualizing VQA Solution for bitstring: {bitstring} ---")

        # Evaluate the bitstring to get a solution dictionary
        solution = self.problem.evaluate_bitstring(bitstring)
        
        # Use the problem's built-in visualization method
        self.problem.visualize_solution(solution)
    