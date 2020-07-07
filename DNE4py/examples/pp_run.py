import sys
import numpy as np

from render import deepga_render, cmaes_render, randomsearch_render


def objective(x):
    result = np.sum(x**2)
    return result


def render_DeepGA_TruncatedRealMutatorGA():
    sigma = 0.05
    num_parents = 3
    num_elite = 1
    deepga_render("results/DeepGA/TruncatedRealMutatorGA", 20, objective, sigma, num_parents, num_elite)


def render_CMAES():
    sigma = 0.05
    cmaes_render("results/CMAES", 20, objective, sigma)


def render_RandomSearch():
    randomsearch_render("results/RandomSearch", 20, objective)


if "__main__" == __name__:

    # Options:
    switcher = {
        0: render_DeepGA_TruncatedRealMutatorGA,
        1: render_CMAES,
        2: render_RandomSearch
    }
    renderize = switcher.get(int(sys.argv[1]), lambda: "Invalid index for optimizer")
    renderize()
