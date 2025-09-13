from pennylane import StronglyEntanglingLayers, BasicEntanglerLayers
import pennylane as qml
from pennylane import numpy as np


# TODO add StronglyEntanglerLayers, BasicEntanglerLayers
# TODO validate Ansatze

def init_params_from_ansatz(vqa, ansatz):
    """
    Initialize a correct parameter tensor according to the ansatz.
    """
    if ansatz == "qaoa_new":
        T = 0.75 * vqa.n_layers
        betas = [-(1 - i/vqa.n_layers) * T / vqa.n_layers for i in range(vqa.n_layers)]
        gammas = [(i / vqa.n_layers) * T / vqa.n_layers for i in range(vqa.n_layers)]
        init_params = np.array([gammas, betas], requires_grad=True)

    elif ansatz == "qaoa":  # TODO BAD INITIALIZATION FOR QAOA!
        init_params = np.random.rand(vqa.n_layers, 2,
                                        requires_grad=True) * 0.02 - 0.01  # 2 parameters (alpha and gamma) for each layer
    elif ansatz == "vqe_barkoutsos":
        init_params = np.random.rand(vqa.n_qubits * (1 + vqa.n_layers),
                                        requires_grad=True) * 0.02 - 0.01  # n * (1 + p) params
    elif ansatz == "vqe_efficient_su2":
        init_params = np.random.rand(vqa.n_qubits * 2 * (1 + vqa.n_layers),
                                        requires_grad=True) * 0.02 - 0.01  # 2n * (1 + p) params
    elif ansatz == "vqe_new":
        init_params = np.random.rand(vqa.n_qubits + vqa.n_layers * (vqa.n_qubits - 1),
                                        requires_grad=True) * 0.02 - 0.01
    else:
        raise ValueError(f"Ansatz {ansatz} not allowed. "
                            f"The ansatz has to be one of : 'qaoa', 'vqe_barkoutsos', "
                            f"'vqe_efficient_su2', 'vqe_new'.")
    return init_params

# FOR OLD METHODS:
def efficient_su2(inputs):
    n_qubits = int(round(inputs.shape[0] / 4, 0))

    # Could iterate over a depth parameter
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
        qml.RZ(inputs[i + n_qubits], wires=i)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
    for i in range(n_qubits):
        qml.RY(inputs[i + 2 * n_qubits], wires=i)
        qml.RZ(inputs[i + 3 * n_qubits], wires=i)


def exp_val(inputs, hamiltonian):
    efficient_su2(inputs)
    return qml.expval(hamiltonian)


def sample(inputs, hamiltonian):
    efficient_su2(inputs)
    return qml.sample(hamiltonian)


def counts(inputs):
    efficient_su2(inputs)
    return qml.measurements.counts()


# NEW VERSIONS - REDUNDANT PARAMETERS FOR WRAPPERS
def efficient_su2_(inputs, n_qubits, layers):
    param_idx = 0

    # First rotation layer
    for i in range(n_qubits):
        qml.RY(inputs[param_idx + i], wires=i)
        qml.RZ(inputs[param_idx + n_qubits + i], wires=i)
    param_idx += 2 * n_qubits

    for layer in range(layers):
        # CNOT layer
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        
        # Rotation layer
        for i in range(n_qubits):
            qml.RY(inputs[param_idx + i], wires=i)
            qml.RZ(inputs[param_idx + n_qubits + i], wires=i)
        param_idx += 2 * n_qubits


def expval_efficient_su2(inputs, n_qubits, layers, hamiltonian):
    efficient_su2_(inputs, n_qubits, layers)
    return qml.expval(hamiltonian)


def sample_efficient_su2(inputs, n_qubits, layers, hamiltonian):
    efficient_su2_(inputs, n_qubits, layers)
    return qml.sample(hamiltonian)


def probs_efficient_su2(inputs, n_qubits, layers, hamiltonian):
    efficient_su2_(inputs, n_qubits, layers)
    return qml.probs()


def counts_efficient_su2(inputs, n_qubits, layers, hamiltonian):
    efficient_su2_(inputs, n_qubits, layers)
    return qml.measurements.counts()


def barkoutsos_vqe(inputs, n_qubits, layers):
    param_idx = 0
    
    # Initial layer of RY rotations
    for i in range(n_qubits):
        qml.RY(inputs[param_idx + i], wires=i)
    param_idx += n_qubits
    qml.Barrier(only_visual=True)

    # Repeat for p layers
    for layer in range(layers):
        # Apply CZ gates between all pairs (i,j) where i < j
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CZ(wires=[i, j])

        # Layer of RY rotations
        for i in range(n_qubits):
            qml.RY(inputs[param_idx + i], wires=i)
        param_idx += n_qubits
        qml.Barrier(only_visual=True)


def barkoutsos_expval(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.expval(hamiltonian)


def barkoutsos_sample(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.sample(hamiltonian)


def barkoutsos_probs(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.probs(hamiltonian)


def barkoutsos_counts(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.measurements.counts()


def barkoutsos_measure(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.measurements.measure(hamiltonian)


def new_vqe(inputs, n_qubits, n_layers):
    """VQE with Ansatz from the Discord image"""

    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)

    param_index = n_qubits  # Start after initial fixed layer

    # Adding the layered block structure as described
    for layer in range(n_layers):
        # First set of CNOTs on even qubits (skips the final qubit if n_qubits is odd)
        for i in range(0, n_qubits - 1, 2):
            qml.CNOT(wires=[i, i + 1])
            qml.RY(inputs[param_index], wires=i)
            qml.RY(inputs[param_index + 1], wires=i + 1)
            param_index += 2

        # Second set of CNOTs on odd qubits (does not skip the final qubit)
        for i in range(1, n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RY(inputs[param_index], wires=i)
            qml.RY(inputs[param_index + 1], wires=i + 1)
            param_index += 2


def new_vqe_sample(inputs, n_qubits, layers, hamiltonian):
    new_vqe(inputs, n_qubits, layers)
    return qml.sample(hamiltonian)


def new_vqe_expval(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.expval(hamiltonian)


def new_vqe_probs(inputs, n_qubits, layers, hamiltonian):
    new_vqe(inputs, n_qubits, layers)
    return qml.probs()


def new_vqe_counts(inputs, n_qubits, layers, hamiltonian):
    barkoutsos_vqe(inputs, n_qubits, layers)
    return qml.measurements.counts()



def qaoa_ansatz(inputs, n_qubits, layers, cost_h):
    # Ground state of the mixer hamiltonian
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    for layer in range(layers):
        qml.qaoa.cost_layer(inputs[layer][0], cost_h)  # The first element of the layer tensor corresponds to h_cost
        # Keep mixer hamiltonian constant: x_mixer
        qml.qaoa.mixer_layer(inputs[layer][1], qml.qaoa.x_mixer(range(n_qubits)))  # The second element of the layer tensor corresponds to h_mixer

def qaoa_sample(inputs, n_qubits, layers, cost_h):
    qaoa_ansatz(inputs, n_qubits, layers, cost_h)
    return qml.sample(cost_h)


def qaoa_probs(inputs, n_qubits, layers, cost_h):
    qaoa_ansatz(inputs, n_qubits, layers, cost_h)
    return qml.probs()


def qaoa_expval(inputs, n_qubits, layers, cost_h):
    qaoa_ansatz(inputs, n_qubits, layers, cost_h)
    return qml.expval(cost_h)


def qaoa_counts(inputs, n_qubits, layers, cost_h):
    qaoa_ansatz(inputs, n_qubits, layers, cost_h)
    return qml.measurements.counts()


def qaoa_new_ansatz(inputs, n_qubits, layers, cost_h, max_cut_bound):
    # Ground state of the mixer hamiltonian
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    for layer in range(layers):
        qml.qaoa.cost_layer(inputs[0][layer], cost_h / max_cut_bound)  # The first element of the layer tensor corresponds to h_cost
        qml.qaoa.mixer_layer(inputs[1][layer], qml.qaoa.x_mixer(range(n_qubits)) / max_cut_bound)  # The second element of the layer tensor corresponds to h_mixer


def qaoa_new_sample(inputs, n_qubits, layers, cost_h, max_cut_bound):
    qaoa_new_ansatz(inputs, n_qubits, layers, cost_h, max_cut_bound)
    return qml.sample(cost_h)


def qaoa_new_probs(inputs, n_qubits, layers, cost_h, max_cut_bound):
    qaoa_new_ansatz(inputs, n_qubits, layers, cost_h, max_cut_bound)
    return qml.probs()


def qaoa_new_expval(inputs, n_qubits, layers, cost_h, max_cut_bound):
    qaoa_new_ansatz(inputs, n_qubits, layers, cost_h, max_cut_bound)
    return qml.expval(cost_h)


def qaoa_new_counts(inputs, n_qubits, layers, cost_h, max_cut_bound):
    qaoa_new_ansatz(inputs, n_qubits, layers, cost_h, max_cut_bound)
    return qml.measurements.counts()


# qml.qaoa.cost_layer(inputs[layer][0], cost_h /)

def hard_coded_cut(bitstring, hamiltonian):
    for i, bit in enumerate(bitstring):
        if bit == '1':
            qml.PauliX(wires=i)
    return qml.expval(hamiltonian)
