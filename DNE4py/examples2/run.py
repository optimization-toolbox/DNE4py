import sys
import numpy as np

from DNE4py.optimizers.deepga2 import TruncatedRealMutatorCompactGA, TruncatedRealMutatorCompositeGA
from DNE4py.optimizers.cmaes import CMAES
from DNE4py.optimizers.random import BatchRandomSearch

# Compact example task
def objective_function(x):
    result = np.sum(x**2)
    return result

def get_deepga_TruncatedRealMutatorCompactGA():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    num_elite = 1
    num_parents = 3
    sigma = 0.05
    seed = 100

    optimizer = TruncatedRealMutatorCompactGA(objective_function,
                                       {'initial_guess': initial_guess,
                                        'workers_per_rank': workers_per_rank,
                                        'num_elite': num_elite,
                                        'num_parents': num_parents,
                                        'sigma': sigma,
                                        'global_seed': seed,
                                        'save': 1,
                                        'verbose': 1,
                                        'output_folder': 'results/DeepGA/TruncatedRealMutatorCompactGA'})
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


# Composite example task
def composite_objective_function(x, n_tasks=50):
    """
    The composite objective is composed of n_tasks functions of the form f_i:(x,y) -> (x^2 + y^2) + i 
    for i in {-n_task/2, ..., n_task/2}.
    Thus mean(f_i) = (x^2 + y^2)
    """
    result = np.array([np.sum(x**2) + i for i in range(-(n_tasks//2), n_tasks//2)])
    return result


def get_deepga_TruncatedRealMutatorCompositeGA():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    num_elite = 1
    num_parents = 3
    sigma = 0.05
    seed = 100

    optimizer = TruncatedRealMutatorCompositeGA(composite_objective_function,
                                       {'initial_guess': initial_guess,
                                        'workers_per_rank': workers_per_rank,
                                        'num_elite': num_elite,
                                        'num_parents': num_parents,
                                        'sigma': sigma,
                                        'global_seed': seed,
                                        'save': 1,
                                        'verbose': 1,
                                        'output_folder': 'results/DeepGA/TruncatedRealMutatorCompositeGA'})
    return optimizer


if "__main__" == __name__:
    # Options:
    switcher = {
        0: get_deepga_TruncatedRealMutatorCompactGA,
        1: get_deepga_TruncatedRealMutatorCompositeGA,
        2: get_cmaes_CMAES,
        3: get_random_BatchRandomSearch,
    }
    optimizer = switcher.get(int(sys.argv[1]), lambda: "Invalid index for optimizer")()
    optimizer.run(20)
