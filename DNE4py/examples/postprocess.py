
import numpy as np

from DNE4py.utils import plot_cost_over_generation, plot_best_cost_over_generation, render_population_over_generation


def objective(x):
    result = np.sum(x**2)
    return result


if "__main__" == __name__:

    plot_cost_over_generation("DNE4py_result", 20)
    plot_best_cost_over_generation("DNE4py_result", 20)

    sigma = 0.05
    num_parents = 3
    num_elite = 1
    render_population_over_generation("DNE4py_result", 20, objective, sigma, num_parents, num_elite)
