import sys
import numpy as np

from DNE4py.optimizers.deepga import TruncatedRealMutatorGA
from DNE4py.optimizers.cmaes import CMAES
from DNE4py.optimizers.random import BatchRandomSearch


def objective_function(x):
    result = np.sum(x**2)
    return result

def get_deepga_TruncatedRealMutatorGA():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    num_elite = 1
    num_parents = 3
    sigma = 0.05
    seed = 100

    optimizer = TruncatedRealMutatorGA(objective_function,
                                       {'initial_guess': initial_guess,
                                        'workers_per_rank': workers_per_rank,
                                        'num_elite': num_elite,
                                        'num_parents': num_parents,
                                        'sigma': sigma,
                                        'global_seed': seed,
                                        'save': 1,
                                        'verbose': 1,
                                        'output_folder': 'results/DeepGA/TruncatedRealMutatorGA'})
    return optimizer

def get_cmaes_CMAES():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    sigma = 0.05
    seed = 100

    optimizer = CMAES(objective_function,
                      {'initial_guess': initial_guess,
                       'workers_per_rank': workers_per_rank,
                       'sigma': sigma,
                       'global_seed': seed,
                       'save': 1,
                       'verbose': 1,
                       'output_folder': 'results/CMAES'})
    return optimizer

def get_random_BatchRandomSearch():

    dim = 2
    workers_per_rank = 10
    bounds = np.array((-1, 1))
    global_seed = 100

    optimizer = BatchRandomSearch(objective_function,
                                  {'dim': dim,
                                   'bounds': bounds,
                                   'workers_per_rank': workers_per_rank,
                                   'global_seed': global_seed,
                                   'save': 1,
                                   'verbose': 1,
                                   'output_folder': 'results/BatchRandomSearch'})
    return optimizer


if "__main__" == __name__:

    # Options:
    switcher = {
        0: get_deepga_TruncatedRealMutatorGA,
        1: get_cmaes_CMAES,
        2: get_random_BatchRandomSearch
    }
    optimizer = switcher.get(int(sys.argv[1]), lambda: "Invalid index for optimizer")()
    optimizer.run(20)
