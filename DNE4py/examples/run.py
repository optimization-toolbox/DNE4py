import sys
import numpy as np

from DNE4py.optimizers.deepga import TruncatedRealMutatorGA
from DNE4py.optimizers.cmaes import CMAES
from DNE4py.optimizers.random_search import RandomSearch


def objective(x):
    result = np.sum(x**2)
    return result

def get_DeepGA_TruncatedRealMutatorGA():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    num_elite = 1
    num_parents = 3
    sigma = 0.05
    seed = 100

    optimizer = TruncatedRealMutatorGA(objective=objective,
                                       initial_guess=initial_guess,
                                       workers_per_rank=workers_per_rank,
                                       num_elite=num_elite,
                                       num_parents=num_parents,
                                       sigma=sigma,
                                       seed=seed,
                                       save=1,
                                       verbose=1,
                                       output_folder='results/DeepGA/TruncatedRealMutatorGA')
    return optimizer

def get_CMAES():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    sigma = 0.05
    seed = 100

    optimizer = CMAES(objective=objective,
                      initial_guess=initial_guess,
                      workers_per_rank=workers_per_rank,
                      sigma=sigma,
                      seed=seed,
                      save=1,
                      verbose=1,
                      output_folder='results/CMAES')
    return optimizer

def get_RandomSearch():

    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank = 10
    bounds = np.array((-1, 1))
    seed = 100

    optimizer = RandomSearch(objective=objective,
                             initial_guess=np.array([-0.3, 0.7]),
                             bounds=bounds,
                             workers_per_rank=workers_per_rank,
                             seed=seed,
                             save=1,
                             verbose=1,
                             output_folder='results/RandomSearch')
    return optimizer


if "__main__" == __name__:

    # Options:
    switcher = {
        0: get_DeepGA_TruncatedRealMutatorGA,
        1: get_CMAES,
        2: get_RandomSearch
    }
    optimizer = switcher.get(int(sys.argv[1]), lambda: "Invalid index for optimizer")()
    optimizer.run(20)
