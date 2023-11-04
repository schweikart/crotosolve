from reconstruction import reconstruct
from minimization import minimize_reconstruction
import pennylane.numpy as np
from numpy.typing import NDArray

class CrotosolveOptimizer:
    def step_and_cost(self, circuit, initial_params: NDArray[np.float_], updates_dataset=[], debug=False):
        """
        reconstruct and optimize the univariate cost functions independently

        TODO: separate CRP params and RP params to save iterations
        """

        prev = circuit(initial_params)
        params = initial_params.copy()

        iterator = np.nditer(params, flags=['multi_index'])
        for old_param_value in iterator:
            param_index = iterator.multi_index
            if debug: print(f"Optimizing parameter {param_index}...")
            univariate = self.create_univariate(circuit, params, param_index)
            reconstruction, constants = reconstruct(univariate)
            new_param_value, new_fun_value = minimize_reconstruction(reconstruction, constants)
            if debug: print(f"Parameter update for {param_index} from {old_param_value} to {new_param_value} ({new_fun_value})")
            params[param_index] = new_param_value
            updates_dataset.append(new_fun_value)
        
        return params, prev

    @staticmethod
    def create_univariate(circuit, params: NDArray[np.int_], param_index):
        def univariate(param_value):
            updated_params = params.copy()
            updated_params[param_index] = param_value
            return circuit(updated_params)
        
        return univariate