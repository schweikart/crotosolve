import pennylane.numpy as np
from numpy.typing import NDArray
from optimizers import OptimizationResult, OptimizationTask
import os
import pickle
from pathlib import Path
from uuid import uuid4

class Instance:
    def __init__(
            self,
            task: OptimizationTask,
            results: dict[str, OptimizationResult],
            uuid: str = None,
    ) -> None:
        self.task = task
        self.results = results
        self.uuid = uuid if uuid is not None else uuid4()
    
    def save(self, folder: str) -> None:
        file = os.path.join(folder, f"{self.task.circuit_id}_{self.task.num_qubits}x{self.task.num_layers}_{self.uuid}.instance")
        pickle.dump(self, open(file, "wb"))
    
    def valid(self) -> bool:
        # TODO: check contained data too!
        return (
            self.task is not None
            and self.results is not None
            and self.uuid is not None
        )

class Dataset:
    def __init__(self, instances: dict[str, Instance] = {}) -> None:
        self.instances = instances

    def readall(self, folder: str) -> None:
        files = os.listdir(folder)
        for filename in files:
            filepath = Path(folder, filename)
            try:
                instance: Instance = pickle.load(open(filepath, "rb"))
            except EOFError:
                print(f"EOFError reading {filename}!")
                continue

            if isinstance(instance, Instance) and instance.valid():
                self.instances[instance.uuid] = instance
            else:
                print(f"Invalid instance {filename}!")
