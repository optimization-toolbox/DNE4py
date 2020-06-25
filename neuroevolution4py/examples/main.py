
import numpy as np
import matplotlib.pyplot as plt

from neuroevolution4py.optimizers.deepga import TruncatedRealMutatorGA
from neuroevolution4py.utils import join_mpidata, load_mpidata
from neuroevolution4py.optimizers.deepga.mutation import Member


def objective(x):
    result = np.sum(x**2)
    return result


def train(steps):
    global initial_guess, sigma

    optimizer = TruncatedRealMutatorGA(objective=objective,
                                       initial_guess=initial_guess,
                                       workers_per_rank=10,
                                       num_elite=1,
                                       num_parents=3,
                                       sigma=sigma,
                                       verbose=True,
                                       seed=2,
                                       save=1
                                       )

    optimizer(steps)

def test(steps):

    global initial_guess, sigma

    # Load weights:
    join_mpidata("costs", nb_workers=2, nb_generations=steps)
    join_mpidata("genotypes", nb_workers=2, nb_generations=steps)
    costs = load_mpidata("costs")
    genotypes = load_mpidata("genotypes")

    # Training performance
    means = np.mean(costs, axis=1)
    stds = np.std(costs, axis=1)
    mins = np.min(costs, axis=1)
    maxes = np.max(costs, axis=1)

    plt.errorbar(np.arange(steps), means, stds, fmt='ok', lw=3)
    plt.errorbar(np.arange(steps), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='gray', lw=1)
    plt.xlim(-1, steps)
    plt.xticks(np.arange(-1, steps + 1, 1.0))
    plt.show()

    # Check same training:
    #new_costs = []
    # for g in range(len(genotypes)):
    #    print(g)
    #    curr_costs = []
    #    for genotype in genotypes[g]:
    #        member = Member(initial_guess, genotype, sigma)
    #        phenotype = member.phenotype
    #        y = objective(phenotype)
    #        curr_costs.append(y)
    #    curr_costs = np.array(curr_costs)
    #    new_costs.append(curr_costs)
    #new_costs = np.array(new_costs)
    #np.save(open("new_costs", "wb"), new_costs)
    #new_costs = np.load("new_costs")


if "__main__" == __name__:

    initial_guess = np.array([1.0, 1.0])
    sigma = 0.1

    # train(20)
    test(20)
