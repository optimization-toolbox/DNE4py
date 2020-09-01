r"""
Abstract base module for all DNE4py optimizers.
"""

from abc import ABC, abstractmethod


class Optimizer(ABC):
    r"""Abstract base class for optimizers."""

    @abstractmethod
    def run(self, objective_function, steps):
        r"""Optimize the objective function.

            Args:
                objective_function: function to be optimized
                steps: Number of iterations
        """
        pass
