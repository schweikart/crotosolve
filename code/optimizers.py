import pennylane as qml
import pennylane.numpy as np
from pennylane import QNode
from numpy.typing import NDArray
from CrotosolveOptimizer import CrotosolveOptimizer
from typing import Callable


class OptimizationTask:
    def __init__(
            self,
            circuit: QNode,
            initial_params: tuple[NDArray[np.float_], NDArray[np.float_]],
            max_evaluations: int = 250,
            convergence_threshold: float = 1e-6,
    ) -> None:
        self.circuit = circuit
        self.max_evaluations = max_evaluations
        self.convergence_threshold = convergence_threshold
        self.initial_params = initial_params

class OptimizationResult:
    def __init__(self, loss: list[tuple[int, float]]) -> None:
        self.loss = loss

def optimize_crotosolve(task: OptimizationTask) -> OptimizationResult:
    max_iterations = task.max_evaluations // (1 + 2 * task.initial_params[0].size + 5 * task.initial_params[1].size)
    optimizer = CrotosolveOptimizer()

    cost = [(0, float(task.circuit(*task.initial_params)))]
    params = task.initial_params
    for iteration in range(max_iterations):
        with qml.Tracker(task.circuit.device) as tracker:
            params, prev_cost, sub_cost = optimizer.step_and_cost(
                task.circuit,
                *params,
                full_output=True
            )
            # assert that the #evaluations estimate is correct
            assert tracker.totals['batches'] == 1 + 2 * params[0].size + 5 * params[1].size

        sub_cost_rp = sub_cost[:params[0].size]
        sub_cost_crp = sub_cost[params[0].size:]

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.extend([
            (
                evaluations_so_far + 1 + 2 * (cost_idx + 1),
                float(cost_value)
            )
            for (cost_idx, cost_value)
            in enumerate(sub_cost_rp)
        ])

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.extend([
            (
                evaluations_so_far + 5 * (cost_idx + 1),
                float(cost_value)
            )
            for (cost_idx, cost_value)
            in enumerate(sub_cost_crp)
        ])

        if np.abs(task.circuit(*params) - prev_cost) <= task.convergence_threshold:
            print("abort", iteration)
            break

    return OptimizationResult(loss=cost)

def optimize_rotosolve(task: OptimizationTask) -> OptimizationResult:
    max_iterations = task.max_evaluations // (3 * task.initial_params[0].size + 5 * task.initial_params[1].size)

    optimizer = qml.RotosolveOptimizer()
    params = task.initial_params

    spectrum_fn = qml.fourier.qnode_spectrum(task.circuit)
    spectra = spectrum_fn(*params)
    cost = [(0, float(task.circuit(*task.initial_params)))]
    for iteration in range(max_iterations):
        with qml.Tracker(task.circuit.device) as tracker:
            params, prev_cost, sub_cost = optimizer.step_and_cost(
                task.circuit,
                *params,
                spectra=spectra,
                full_output=True
            )
            assert tracker.totals['batches'] == 3 * params[0].size + 5 * params[1].size

        sub_cost_rp = sub_cost[:params[0].size]
        sub_cost_crp = sub_cost[params[0].size:]

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.extend([
            (
                evaluations_so_far + 3 * (cost_idx + 1),
                float(cost_value)
            )
            for (cost_idx, cost_value)
            in enumerate(sub_cost_rp)
        ])

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.extend([
            (
                evaluations_so_far + 5 * (cost_idx + 1),
                float(cost_value)
            )
            for (cost_idx, cost_value)
            in enumerate(sub_cost_crp)
        ])

        if np.abs(task.circuit(*params) - prev_cost) <= task.convergence_threshold:
            print("abort", iteration)
            break

    return OptimizationResult(loss=cost)

def optimize_gradientdescent(task: OptimizationTask) -> OptimizationResult:
    max_iterations = task.max_evaluations // 2 # 2 evals per iteration

    optimizer = qml.GradientDescentOptimizer()
    params = task.initial_params

    cost = [(0, float(task.circuit(*task.initial_params)))]
    for iteration in range(max_iterations):
        with qml.Tracker(task.circuit.device) as tracker:
            params, prev_cost = optimizer.step_and_cost(
                task.circuit,
                *params,
            )
        evaluations_here = tracker.totals['batches']
        assert evaluations_here == 2, "Gradient needs two evaluations!"
        current_cost = float(task.circuit(*params))

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.append((evaluations_so_far + evaluations_here, current_cost))

        if np.abs(current_cost - prev_cost) <= task.convergence_threshold:
            print("abort", iteration)
            break
        if iteration % 20 == 0:
            print(iteration, current_cost)

    return OptimizationResult(loss=cost)

def optimize_adam(task: OptimizationTask) -> OptimizationResult:
    max_iterations = task.max_evaluations // 2 # 2 evals per iteration
    optimizer = qml.AdamOptimizer()
    params = task.initial_params

    cost = [(0, float(task.circuit(*task.initial_params)))]
    for iteration in range(max_iterations):
        with qml.Tracker(task.circuit.device) as tracker:
            params, prev_cost = optimizer.step_and_cost(
                task.circuit,
                *params,
            )
        evaluations_here = tracker.totals['batches']
        assert evaluations_here == 2, "Gradient needs two evaluations!"
        current_cost = float(task.circuit(*params))

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.append((evaluations_so_far + evaluations_here, current_cost))

        if np.abs(task.circuit(*params) - prev_cost) <= task.convergence_threshold:
            print("abort", iteration)
            break
        if iteration % 20 == 0:
            print(iteration, current_cost)

    return OptimizationResult(loss=cost)

def optimize_adagrad(task: OptimizationTask) -> OptimizationResult:
    max_iterations = task.max_evaluations // 2 # 2 evals per iteration
    optimizer = qml.AdagradOptimizer()
    params = task.initial_params

    cost = [(0, float(task.circuit(*task.initial_params)))]
    for iteration in range(max_iterations):
        with qml.Tracker(task.circuit.device) as tracker:
            params, prev_cost = optimizer.step_and_cost(
                task.circuit,
                *params,
            )
        evaluations_here = tracker.totals['batches']
        assert evaluations_here >= 2, "Gradient needs two evaluations!"
        current_cost = float(task.circuit(*params))

        evaluations_so_far = max([evs for (evs, _) in cost], default=0)
        cost.append((evaluations_so_far + evaluations_here, current_cost))

        if np.abs(task.circuit(*params) - prev_cost) <= task.convergence_threshold:
            print("abort", iteration)
            break
        if iteration % 20 == 0:
            print(iteration, current_cost)

    return OptimizationResult(loss=cost)

Optimizer = Callable[[OptimizationTask], OptimizationResult]

optimizers: list[tuple[str, Optimizer]] = [
    ("Crotosolve", optimize_crotosolve),
    ("Rotosolve", optimize_rotosolve),
    ("Gradient Descent", optimize_gradientdescent),
    ("Adam", optimize_adam),
    ("Adagrad", optimize_adagrad)
]