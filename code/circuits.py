import pennylane as qml
import pennylane.numpy as np
from numpy.typing import NDArray

def sim_03(num_layers: int = 5, num_qubits: int = 4):
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        """
        rp_params is a three-dimensional array (#layers, #qubits, 2) where
        rp_params[l, q] are the RX and RZ parameters for the q-th qubit in the l-th layer.

        crp_params is a two-dimensional array (#layers, #qubits - 1) where
        crp_params[l, q] is the CRZ parameter for the q-th qubit in the l-th layer
        """
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                qml.RX(rp_params[layer, qubit, 0], wires=qubit)
                qml.RZ(rp_params[layer, qubit, 1], wires=qubit)
            for qubit in range(num_qubits - 2, -1, -1): # incl. start, excl. stop, step
                qml.CRZ(crp_params[layer, qubit], (qubit + 1, qubit))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(1))

    return circuit
