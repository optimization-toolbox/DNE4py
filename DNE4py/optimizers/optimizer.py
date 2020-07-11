r"""
Abstract base module for all DNE4py optimizers.
"""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    r"""Abstract base class for optimizers."""

    def __init__(self, objective_function, config):
        r"""Constructor for the Optimizer base class.

            Args:
                objective_function: callable function in python
                config: configurable setting for the optimizer

        """
        self.objective_function = objective_function
        self.config = config
        self.__dict__.update(config)

    @abstractmethod
    def run(self, steps):
        r"""Optimize the objective function.

            Args:
                steps: Number of iterations
        """
        pass
