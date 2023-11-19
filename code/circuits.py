import pennylane as qml
import pennylane.numpy as np
from numpy.typing import NDArray

default_num_layers = 5
default_num_qubits = 4

def default_device(num_qubits: int) -> qml.Device:
    return qml.device("default.qubit", wires=num_qubits)

def _R_layer(r_gate, rp_params: NDArray[np.float_], num_qubits: int):
    for qubit in range(num_qubits):
        r_gate(rp_params[qubit], wires=qubit)

def sim_01(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)
            qml.Barrier(only_visual=True)
        
        return qml.expval(qml.PauliZ(0))
    
    return circuit

def sim_02(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RY, rp_params[layer, :, 1], num_qubits)

            for qubit in range(num_qubits - 1, 0, -1):
                qml.CNOT(wires=(qubit, qubit - 1))

            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    
    return circuit

def sim_03(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        """
        rp_params is a three-dimensional array (#layers, #qubits, 2) where
        rp_params[l, q] are the RX and RZ parameters for the q-th qubit in the l-th layer.

        crp_params is a two-dimensional array (#layers, #qubits - 1) where
        crp_params[l, q] is the CRZ parameter for the q-th qubit in the l-th layer
        """
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for qubit in range(num_qubits - 2, -1, -1): # incl. start, excl. stop, step
                qml.CRZ(crp_params[layer, qubit], (qubit + 1, qubit))

            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

def sim_04(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        """
        rp_params is a three-dimensional array (#layers, #qubits, 2) where
        rp_params[l, q] are the RX and RZ parameters for the q-th qubit in the l-th layer.

        crp_params is a two-dimensional array (#layers, #qubits - 1) where
        crp_params[l, q] is the CRX parameter for the q-th qubit in the l-th layer
        """
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for qubit in range(num_qubits - 2, -1, -1): # incl. start, excl. stop, step
                qml.CRX(crp_params[layer, qubit], (qubit + 1, qubit))
                
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

def _full_stairs(cgate, stairs_params: NDArray[np.float_], num_qubits: int = default_num_qubits):
    for control in range(num_qubits - 1, -1, -1):
        for target in range(num_qubits - 1, -1, -1):
            if control != target:
                param_idx = num_qubits - 1 - control
                if target > control:
                    param_idx -= 1
                cgate(stairs_params[target, param_idx], (control, target))

def sim_05(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            _full_stairs(qml.CRZ, crp_params[layer], num_qubits)
            
            _R_layer(qml.RX, rp_params[layer, :, 2], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 3], num_qubits)
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

def sim_06(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            _full_stairs(qml.CRX, crp_params[layer], num_qubits)
            
            _R_layer(qml.RX, rp_params[layer, :, 2], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 3], num_qubits)
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

def _odd_controlled_gate_wires(num_qubits: int = default_num_qubits):
    return [
        (control, control - 1) # target is always above control
        for control in range(1, num_qubits, 2) # note: control >= 1
    ]

def _even_controlled_gate_wires(num_qubits: int = default_num_qubits):
    return [
        (control, control - 1) # target is always above control
        for control in range(2, num_qubits, 2) # note: control >= 2
    ]

def _odd_crps(gate, crp_params: NDArray[np.float_], num_qubits: int = default_num_qubits):
    for (control, target) in _odd_controlled_gate_wires(num_qubits):
        gate(crp_params[control - 1], wires=(control, target))

def _even_crps(gate, crp_params: NDArray[np.float_], num_qubits: int = default_num_qubits):
    for (control, target) in _even_controlled_gate_wires(num_qubits):
        gate(crp_params[control - 1], wires=(control, target))

def sim_07(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _odd_controlled_gate_wires(num_qubits):
                qml.CRZ(crp_params[layer, target], wires=(control, target))

            _R_layer(qml.RX, rp_params[layer, :, 2], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 3], num_qubits)

            for (control, target) in _even_controlled_gate_wires(num_qubits):
                qml.CRZ(crp_params[layer, target], wires=(control, target))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    
    return circuit

def sim_08(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _odd_controlled_gate_wires(num_qubits):
                qml.CRX(crp_params[layer, target], wires=(control, target))

            _R_layer(qml.RX, rp_params[layer, :, 2], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 3], num_qubits)

            for (control, target) in _even_controlled_gate_wires(num_qubits):
                qml.CRX(crp_params[layer, target], wires=(control, target))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    
    return circuit

def sim_09(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            for qubit in range(num_qubits):
                qml.Hadamard(wires=qubit)
            
            for control in range(num_qubits - 1, 0, -1):
                target = (control - 1 + num_qubits) % num_qubits
                qml.CZ(wires=(control, target))
            
            _R_layer(qml.RX, rp_params[layer], num_qubits)
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    
    return circuit

def sim_10(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for qubit in range(num_qubits):
            qml.RY(rp_params[0, qubit], wires=qubit)

        for layer in range(num_layers):
            for control in range(num_qubits - 1, -1, -1):
                target = (control - 1 + num_qubits) % num_qubits
                qml.CZ(wires=(control, target))
            
            _R_layer(qml.RY, rp_params[layer + 1], num_qubits)
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    
    return circuit

def sim_11(num_layers: int = default_num_layers) -> qml.QNode:
    num_qubits = 4 # no idea how to generalize this
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RY, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _odd_controlled_gate_wires(num_qubits):
                qml.CNOT(wires=(control, target))
            
            qml.RY(rp_params[layer, 1, 2], wires=1)
            qml.RY(rp_params[layer, 2, 2], wires=2)

            # no idea how to do this any better
            qml.RY(rp_params[layer, 0, 2], wires=1)
            qml.RY(rp_params[layer, 3, 2], wires=2)

            qml.CNOT(wires=(2, 1))
            qml.Barrier(only_visual=True)

        return qml.expval(qml.PauliZ(0))

    return circuit

def sim_12(num_layers: int = default_num_layers) -> qml.QNode:
    num_qubits = 4 # no idea how to generalize this
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RY, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _odd_controlled_gate_wires(num_qubits):
                qml.CZ(wires=(control, target))
            
            qml.RY(rp_params[layer, 1, 2], wires=1)
            qml.RY(rp_params[layer, 2, 2], wires=2)

            # no idea how to do this any better
            qml.RY(rp_params[layer, 0, 2], wires=1)
            qml.RY(rp_params[layer, 3, 2], wires=2)

            qml.CZ(wires=(2, 1))
            qml.Barrier(only_visual=True)

        return qml.expval(qml.PauliZ(0))

    return circuit

def _stairs(start: int = 0, num_qubits: int = default_num_qubits, direction: int = 1, target_offset: int = 1):
    return [
        (
            control % num_qubits,
            (control + target_offset) % num_qubits
        )
        for control in range(start, start + direction * num_qubits, direction)
    ]

def sim_13(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RY, rp_params[layer, :, 0], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=-1, target_offset=+1):
                qml.CRZ(crp_params[layer, target, 0], wires=(control, target))
            
            _R_layer(qml.RY, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=+1, target_offset=-1):
                qml.CRZ(crp_params[layer, target, 1], wires=(control, target))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

def sim_14(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits):
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RY, rp_params[layer, :, 0], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=-1, target_offset=+1):
                qml.CRX(crp_params[layer, target, 0], wires=(control, target))
            
            _R_layer(qml.RY, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=+1, target_offset=-1):
                qml.CRX(crp_params[layer, target, 1], wires=(control, target))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))

    return circuit

def sim_15(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RY, rp_params[layer, :, 0], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=-1, target_offset=+1):
                qml.CNOT(wires=(control, target))
            
            _R_layer(qml.RY, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=+1, target_offset=-1):
                qml.CNOT(wires=(control, target))
            
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    return circuit

def sim_16(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _odd_controlled_gate_wires(num_qubits):
                qml.CRZ(crp_params[layer, target], wires=(control, target))
            for (control, target) in _even_controlled_gate_wires(num_qubits):
                qml.CRZ(crp_params[layer, target], wires=(control, target))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    return circuit

def sim_17(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _odd_controlled_gate_wires(num_qubits):
                qml.CRX(crp_params[layer, target], wires=(control, target))
            for (control, target) in _even_controlled_gate_wires(num_qubits):
                qml.CRX(crp_params[layer, target], wires=(control, target))
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    return circuit

def sim_18(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=-1, target_offset=+1):
                qml.CRZ(crp_params[layer, target], wires=(control, target))
            
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    return circuit

def sim_19(num_layers: int = default_num_layers, num_qubits: int = default_num_qubits) -> qml.QNode:
    dev = default_device(num_qubits)

    @qml.qnode(dev)
    def circuit(rp_params: NDArray[np.float_], crp_params: NDArray[np.float_]):
        for layer in range(num_layers):
            _R_layer(qml.RX, rp_params[layer, :, 0], num_qubits)
            _R_layer(qml.RZ, rp_params[layer, :, 1], num_qubits)

            for (control, target) in _stairs(start=num_qubits-1, num_qubits=num_qubits, direction=-1, target_offset=+1):
                qml.CRX(crp_params[layer, target], wires=(control, target))
            
            qml.Barrier(only_visual=True)
        return qml.expval(qml.PauliZ(0))
    return circuit