from reconstruction import reconstruct
from minimization import minimize_reconstruction
import pennylane.numpy as np
from numpy.typing import NDArray
from pennylane import QNode
from typing import Callable

class CrotosolveOptimizer:
    def step_and_cost(self, circuit, initial_rp_params: NDArray[np.float_], initial_crp_params: NDArray[np.float_], updates_dataset: list[float] = [], debug=False, full_output=False):
        """
        reconstruct and optimize the univariate cost functions independently
        """

        prev = circuit(initial_rp_params, initial_crp_params)
        y_output = []

        rp_params = initial_rp_params.copy()
        crp_params = initial_crp_params.copy()

        # by caching the final value after each step, we can save #steps evaluations!
        cache = prev

        for params, gate in [(rp_params, "RP"), (crp_params, "CRP")]:
            iterator = np.nditer(params, flags=['multi_index', 'zerosize_ok'])
            for old_param_value in iterator:
                param_index = iterator.multi_index
                if debug: print(f"Optimizing {gate} parameter {param_index}...")

                univariate = self._create_univariate(circuit, rp_params, crp_params, param_index, gate)
                reconstruction, constants = reconstruct(univariate, theta=old_param_value, value_at_theta=cache, gate=gate)
                new_param_value, new_fun_value = minimize_reconstruction(reconstruction, constants) # TODO gate!

                if debug: print(f"{gate} parameter update for {param_index} from {old_param_value} to {new_param_value} -> y = {new_fun_value}")
                params[param_index] = new_param_value
                updates_dataset.append(new_fun_value)
                y_output.append(new_fun_value)

                cache = new_fun_value

        if full_output:
            return (rp_params, crp_params), prev, y_output
        
        return (rp_params, crp_params), prev

    @staticmethod
    def _create_univariate(
        circuit: QNode,
        rp_params: NDArray[np.float_],
        crp_params: NDArray[np.float_],
        param_index: tuple,
        gate: str,
    ) -> Callable:
        if gate == "RP":
            def univariate(param_value):
                updated_rp_params = rp_params.copy()
                updated_rp_params[param_index] = param_value
                return circuit(updated_rp_params, crp_params)
            
            return univariate
        else:
            def univariate(param_value):
                updated_crp_params = crp_params.copy()
                updated_crp_params[param_index] = param_value
                return circuit(rp_params, updated_crp_params)
            
            return univariate
