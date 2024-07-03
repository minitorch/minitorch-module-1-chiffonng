from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Dict, Iterable, List, Set, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(
    f: Callable[..., Any], *vals: Any, arg: int = 0, epsilon: float = 1e-6
) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Central difference formula:
    $f'(x) = \frac{f(x + h) - f(x - h)}{2h}$
    Here $h$ is a small constant, and $f(x)$ is the function we want to differentiate.

    To compute f'(x) with respect to the i-th argument, we perturb/vary the i-th argument by a small constant h, and compute the difference in the function values.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    # Calculate x - h
    left_vals = list(vals)
    left_vals[arg] -= epsilon

    # Calculate x + h
    right_vals = list(vals)
    right_vals[arg] += epsilon

    # Calculate f'(x) = (f(x + h) - f(x - h)) / 2h
    return (f(*right_vals) - f(*left_vals)) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Topological order is a linear ordering of its nodes such that for every directed edge (a, b), 'a' comes before 'b'. All dependencies of a node therefore are evaluated before the node itself.

    Time complexity: O(V + E), where V is the number of variables and E is the number of edges.
    Space complexity: O(V), since all data structures are linear in the number of variables (in_degree, stack, visited, result)

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # Dictionary to store number of incoming edges.
    # {variable_id: in_degree_count}. Default value is 0
    in_degree: Dict[int, int] = defaultdict(int)
    in_degree[variable.unique_id] = 0

    # Stack to keep track of nodes to visit, using doubly ended queue for O(1) pop and append
    stack: Deque[Variable] = deque([variable])
    visited: Set[int] = set([variable.unique_id])  # Keep track of visited nodes
    result: List[Variable] = []  # List to store the topological order

    # First pass: Calculate in-degrees and identify all nodes, using iterative DFS
    while stack:
        cur_var = stack.pop()

        # Explore the parents of the current variable, counting the incoming edges
        for var in cur_var.parents:
            # Skip constant variables since they do not have derivatives
            # Otherwise, increment the in-degree of the parent
            if not var.is_constant():
                in_degree[var.unique_id] += 1

                # If the parent has not been visited, add it to the stack
                if var.unique_id not in visited:
                    stack.append(var)
                    visited.add(var.unique_id)

    # Reset the stack and add the variable to the stack
    stack.append(variable)

    # Second pass: Topological sorting using zero in-degree nodes
    # Only add variable to the result when all its dependencies (i.e. parents) have been processed (in_degree = 0)
    while stack:
        cur_var = stack.pop()
        result.append(cur_var)

        for var in cur_var.parents:
            # If the variable is not a constant, decrement the number of incoming edges because the parent will be visited
            if not var.is_constant():
                in_degree[var.unique_id] -= 1

                # If the parent has zero incoming edges, add it to the stack to be visited
                if in_degree[var.unique_id] == 0:
                    stack.append(var)

    return result


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Dictionary to store derivatives for each variable
    derivatives_dict = {variable.unique_id: deriv}

    top_sort = topological_sort(variable)

    # Iterate through the topological order and calculate the derivatives
    for curr_var in top_sort:
        if curr_var.is_leaf():
            continue

        # Get the derivatives of the current variable
        var_n_der = curr_var.chain_rule(derivatives_dict[curr_var.unique_id])

        # Accumulate the derivative for each parent of the current variable
        for var, deriv in var_n_der:
            if var.is_leaf():
                var.accumulate_derivative(deriv)
            else:
                if var.unique_id not in derivatives_dict:
                    derivatives_dict[var.unique_id] = deriv
                else:
                    derivatives_dict[var.unique_id] += deriv


@dataclass
class Context:
    """
    Context class is visited by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be visited during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
