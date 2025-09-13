
from qiskit_circuits import QuantumCircuit
from qiskit_circuits.circuit.library import TwoLocal
from qiskit_circuits.primitives import StatevectorEstimator as Estimator


def two_local(num_qubits):
    reference_circuit = QuantumCircuit(num_qubits)
    # reference_circuit.x(0)  Might add initialization gates such as X gates or Hadammard gates

    variational_form = TwoLocal(
        num_qubits,
        rotation_blocks=["rz", "ry"],
        entanglement_blocks="cx",
        entanglement="circular",
        reps=1,
    )

    ansatz = reference_circuit.compose(variational_form)
    return ansatz


def cost(params, ansatz, hamiltonian):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    estimator = Estimator()

    # job = estimator.run([(ansatz, hamiltonian, params)])

    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    # cost_history_dict["iters"] += 1
    # cost_history_dict["prev_vector"] = params
    # cost_history_dict["cost_history"].append(energy)
    # print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy
