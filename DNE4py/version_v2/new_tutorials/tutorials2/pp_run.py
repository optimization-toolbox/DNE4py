import sys
import numpy as np

from render import deepga_render, composite_deepga_render, cmaes_render, randomsearch_render


def objective(x):
    result = np.sum(x**2)
    return result

def render_Compact_DeepGA():
    sigma = 0.05
    num_parents = 3
    num_elite = 1
    deepga_render("results/DeepGA/TruncatedRealMutatorCompactGA", 20, objective, sigma, num_parents, num_elite)

def render_Composite_DeepGA():

    n_tasks = 4
    sigma = 0.05
    num_parents = 3
    num_elite = 1
    composite_deepga_render("results/DeepGA/TruncatedRealMutatorCompositeGA", 20, objective, sigma, num_parents, num_elite)

def render_CMAES():
    sigma = 0.05
    cmaes_render("results/CMAES", 20, objective, sigma)

def render_RandomSearch():
    randomsearch_render("results/RandomSearch", 20, objective)

# Composite example task
# def multi_objective(x, n_tasks = 4):
#     """
#     The composite objective is composed of n_tasks functions of the form f:(x,y) -> (x^2 + y^2) * i
#     The sum of the component individual objectives is used to evaluate the optimizer 
#     (we expect it to optimize the average)  
#     """
#     result = np.array([np.sum(x**2 + i*x ) * i for i in range(n_tasks)])
#     return sum(result)


if "__main__" == __name__:

    # Options:
    switcher = {
        0: render_Compact_DeepGA,
        1: render_Composite_DeepGA,
        2: render_CMAES,
        3: render_RandomSearch, 
    }
    renderize = switcher.get(int(sys.argv[1]), lambda: "Invalid index for optimizer")
    renderize()
