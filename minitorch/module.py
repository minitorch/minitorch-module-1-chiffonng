from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class Module:
    """
    Modules form a tree that store parameters and other
    submodules. They make up the basis of neural network stacks.

    Attributes:
        _modules : Storage of the child modules
        _parameters : Storage of the module's parameters
        training : Whether the module is in training mode or evaluation mode

    """

    _modules: Dict[str, Module]
    _parameters: Dict[str, Parameter]
    training: bool

    def __init__(self) -> None:
        self._modules = {}
        self._parameters = {}
        self.training = True

    def modules(self) -> Sequence[Module]:
        "Return the direct child modules of this module."
        m: Dict[str, Module] = self.__dict__["_modules"]
        return list(m.values())

    def train(self) -> None:
        "Set the mode of this module and all descendent modules to `train`."
        self.training = True

        # Create a stack and add the current module to it
        stack = list(self.modules())

        # Loop until the stack is empty
        while stack:
            module = stack.pop()

            # Set the module to training mode
            module.training = True

            # Add the descendant modules of the current module to the stack
            stack.extend(module.modules())

    def eval(self) -> None:
        "Set the mode of this module and all descendent modules to `eval`."
        self.training = False

        # Create a stack and add the current module to it
        stack = list(self.modules())

        # Loop until the stack is empty
        while stack:
            module = stack.pop()

            # Set the module to evaluation mode
            module.training = False

            # Add the descendant modules of the current module to the stack
            stack.extend(module.modules())

    def named_parameters(self) -> List[Tuple[str, Parameter]]:
        """
        Collect all the parameters of this module and its descendants.
        Implemntation: Iterative, preorder traversal of the module tree.

        Returns:
            The name and `Parameter` of each ancestor parameter.
        """
        # Initialize an empty list to store named parameters
        named_params: List[Tuple[str, Parameter]] = []

        # Initialize a stack list for DFS traversal
        stack: List[Tuple[str, Module]] = []

        # Add the root module with an empty name to start the traversal
        stack.append(("", self))

        while len(stack) != 0:  # While there are modules to traverse
            name, tail = stack.pop()  # Pop a module from the stack

            # Get the child modules and parameters of the current module
            module_without_name = list(tail.__dict__["_modules"].items())
            item_without_module_key = list(tail.__dict__["_parameters"].items())

            # If the current module has no name (root module)
            if name == "":
                # Add child modules and parameters to the stack and named_params list
                stack.extend(module_without_name)
                named_params.extend(item_without_module_key)
                continue

            # If the current module has a name
            # Add child modules with updated names (including parent module names) to the stack
            stack.extend([(name + "." + key, val) for key, val in module_without_name])

            # Add parameters with updated names (including parent module names) to the named_params list
            named_params.extend(
                [(name + "." + key, val) for key, val in item_without_module_key]
            )

        return named_params  # Return the list of named parameters

    def parameters(self) -> List[Parameter]:
        "Enumerate over all the parameters of this module and its descendents."
        params: List[Parameter] = []
        stack: List[Module] = [self]  # Initialize a stack with the root module

        while stack:  # While there are modules to explore
            current_module = stack.pop()

            # Collect parameters of the current module
            params.extend(current_module._parameters.values())

            # Add child modules to the stack for exploration
            stack.extend(current_module._modules.values())

        return params

    def add_parameter(self, k: str, v: Any) -> Parameter:
        """
        Manually add a parameter. Useful helper for scalar parameters.

        Args:
            k: Local name of the parameter.
            v: Value for the parameter.

        Returns:
            Newly created parameter.
        """
        val = Parameter(v, k)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key: str, val: Parameter) -> None:
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key: str) -> Any:
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        def _addindent(s_: str, numSpaces: int) -> str:
            s2 = s_.split("\n")
            if len(s2) == 1:
                return s_
            first = s2.pop(0)
            s2 = [(numSpaces * " ") + line for line in s2]
            s = "\n".join(s2)
            s = first + "\n" + s
            return s

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.

    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x: Any, name: Optional[str] = None) -> None:
        self.value = x
        self.name = name
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def update(self, x: Any) -> None:
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)
            if self.name:
                self.value.name = self.name

    def __repr__(self) -> str:
        return repr(self.value)

    def __str__(self) -> str:
        return str(self.value)
