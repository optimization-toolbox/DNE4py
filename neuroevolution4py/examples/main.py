
import numpy as np
import matplotlib.pyplot as plt

from neuroevolution4py.optimizers.deepga import TruncatedRealMutatorGA
from neuroevolution4py.utils import join_mpidata, load_mpidata
from neuroevolution4py.optimizers.deepga.mutation import Member


def objective(x):

    result = np.sum(x**2)

    x1 = x[0].round(10)
    x2 = x[1].round(10)
    y = result.round(10)
    # print(f'function: x = [{x1},{x2}], f = {y}')

    return result


def train(steps):
    global initial_guess, sigma

    optimizer = TruncatedRealMutatorGA(objective=objective,
                                       initial_guess=initial_guess,
                                       workers_per_rank=128,
                                       num_elite=1,
                                       num_parents=10,
                                       sigma=sigma,
                                       verbose=True,
                                       seed=2,
                                       save=1
                                       )

    optimizer(steps)

def test(steps):

    global initial_guess, sigma

    # Load weights:
    #join_mpidata("costs", nb_workers=2, nb_generations=steps)
    #join_mpidata("genotypes", nb_workers=2, nb_generations=steps)
    costs = load_mpidata("costs")
    genotypes = load_mpidata("genotypes")

    # Training performance
    #means = np.mean(costs, axis=1)
    #stds = np.std(costs, axis=1)
    #mins = np.min(costs, axis=1)
    #maxes = np.max(costs, axis=1)

    #plt.errorbar(np.arange(steps), means, stds, fmt='ok', lw=3)
    # plt.errorbar(np.arange(steps), means, [means - mins, maxes - means],
    #             fmt='.k', ecolor='gray', lw=1)
    #plt.xlim(-1, steps)
    #plt.xticks(np.arange(-1, steps + 1, 1.0))
    # plt.show()

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
    new_costs = np.load("new_costs")
    a = np.sort(costs)
    b = np.sort(new_costs)
    print(a[1])
    print(b[1])
    print(np.equal(a[1], b[1]))
    exit()

    print(np.equal(new_costs[1], costs[1]))
    print(new_costs[1])
    print("\n\n")
    print(costs[1])

    # print(costs)
    # print(genotypes)

#    for g in range(len(genotypes)):

#        # print(f"Generation: {g}")
#        population_genotype = genotypes[g]
#        print(population_genotype)
#        print("\n\n")
#        continue

#        for genotype in population_genotype:
#            #print(initial_guess, sigma, genotype)
#            member = Member(initial_guess, genotype, sigma)
#            phenotype = member.phenotype

#            x1 = phenotype[0].round(10)
#            x2 = phenotype[1].round(10)
#            y = objective(phenotype).round(10)
#            print(f'load: [{x1},{x2}], {y}')
#        print("\n")


if "__main__" == __name__:

    initial_guess = np.array([1.0, 1.0])
    sigma = 0.1

    # train(100)
    test(100)
