import pennylane as qml
import pennylane.numpy as np
from numpy.typing import NDArray

def sim_03(num_layers: int = 5, num_qubits: int = 4):
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(params: NDArray[np.float_]):
        """
        params is a three-dimensional array where
        params[l, q] are the three parameters for the q-th qubit in the l-th layer.
        """
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                qml.RX(params[layer, qubit, 0], wires=qubit)
                qml.RZ(params[layer, qubit, 1], wires=qubit)
            for qubit in range(num_qubits - 2, -1, -1): # incl. start, excl. stop, step
                qml.CRZ(params[layer, qubit, 2], (qubit + 1, qubit))
        return qml.expval(qml.PauliZ(1))

    return circuit
