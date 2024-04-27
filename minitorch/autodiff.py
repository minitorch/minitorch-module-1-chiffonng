from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Tuple

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

    Raises:
        ValueError: if the function and input values are not provided, or if the argument index is out of range.
    """
    # Ensure the function has at least one argument
    if not callable(f) or not vals:
        raise ValueError("The function and input values must be provided.")

    # Ensure the argument index is within the range of the input values
    if arg < 0 or arg >= len(vals):
        raise ValueError(
            "The argument index must be smaller than the range of the input values."
        )

    # vals is a tuple, convert to list to modify the values
    val_lst: List[Any] = list(vals)

    # Calculate f(x + h)
    val_lst[arg] += epsilon
    f_added: Any = f(*val_lst)

    # Calculate f(x - h)
    val_lst[arg] -= 2 * epsilon
    f_minus: Any = f(*val_lst)

    # Calculate f'(x) = (f(x + h) - f(x - h)) / 2h
    derivative_approx: float = (f_added - f_minus) / (2 * epsilon)

    return derivative_approx


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

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    visited = set()  # To keep track of visited nodes

    # Stack-based iterative depth-first search
    stack = [variable]
    reverse_order: List[Variable] = []

    while stack:
        current = stack.pop()

        # If the node is not visited, visit it
        if current.unique_id not in visited:
            visited.add(current)

        # If the node is not a constant, explore its parents
        if not current.is_constant():
            for parent in current.parents:
                stack.append(parent)

        reverse_order.append(current)
    reverse_order.reverse()

    return reverse_order


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leaf nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Initialize the stack with the variable and its derivative
    stack: List[Tuple[Variable, Any]] = [(variable, deriv)]

    # Traverse the graph in topological order
    for var in topological_sort(variable):
        # If the variable is a leaf, accumulate the derivative
        if var.is_leaf():
            var.accumulate_derivative(deriv)

        # If the variable is not a leaf, compute the chain rule
        else:
            # Get the derivative from the stack
            var_deriv = 0
            for parent, parent_deriv in stack:
                if var in [p for p, _ in parent.chain_rule(parent_deriv)]:
                    var_deriv += parent_deriv

            # Compute the chain rule for the variable
            for parent, parent_deriv in var.chain_rule(var_deriv):
                stack.append((parent, parent_deriv))


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
