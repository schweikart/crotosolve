import pennylane.numpy as np
from numpy.typing import NDArray
from optimizers import OptimizationResult
import os
import pickle
from pathlib import Path

class Instance:
    def __init__(
            self,
            circuit_name: str,
            num_qubits: int,
            num_layers: int,
            initial_params: tuple[NDArray[np.float_], NDArray[np.float_]],
            results: dict[str, OptimizationResult]
    ) -> None:
        self.circuit_name = circuit_name
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.initial_params = initial_params
        self.results = results

class Dataset:
    def __init__(self, instances: dict[str, Instance] = {}) -> None:
        self.instances = instances

    def readall(self, folder: str) -> None:
        files = os.listdir(folder)
        for filename in files:
            filepath = Path(folder, filename)
            instance = pickle.load(open(filepath, "rb"))
            self.instances[filepath.stem] = instance