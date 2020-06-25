import numpy as np

from DNE4py.optimizers.deepga import TruncatedRealMutatorGA


def objective(x):
    result = np.sum(x**2)
    return result


if "__main__" == __name__:

    initial_guess = np.array([-0.3, 0.7])
    sigma = 0.05

    optimizer = TruncatedRealMutatorGA(objective=objective,
                                       initial_guess=initial_guess,
                                       workers_per_rank=10,
                                       num_elite=1,
                                       num_parents=3,
                                       sigma=sigma,
                                       seed=100,
                                       save=1,
                                       verbose=1,
                                       output_folder='DNE4py_result')

    optimizer.run(20)
