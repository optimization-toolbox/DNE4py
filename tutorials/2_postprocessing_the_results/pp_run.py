import sys
import numpy as np

from DNE4py import load_mpidata, get_best_phenotype, get_best_phenotype_generator
import matplotlib.pyplot as plt


def objective_function(x):
    return np.sum(x**2)


if "__main__" == __name__:

    costs = load_mpidata('costs', 'results/TruncatedRealMutatorGA/')
    genotypes = load_mpidata('genotypes', 'results/TruncatedRealMutatorGA/')
    initial_guess = load_mpidata('initial_guess', 'results/TruncatedRealMutatorGA/')

    # for i, c in enumerate(costs):
    #     print(i)
    #     print(f'{np.min(c)}')
    #     print(f'{c}')
    #     print()

    print(f'costs shape: {costs.shape}')
    print(f'genotypes shape: {genotypes.shape}')
    print(f'initial_guess shape: {initial_guess.shape}')

    print(genotypes[0])
    print(genotypes[-1])
    #print(initial_guess)
    #print(costs[-1][0])

    x = get_best_phenotype('results/TruncatedRealMutatorGA/')
    
    #exit()
    #print(x)
    #print(objective_function(x))
    #exit()

    generator = get_best_phenotype_generator('results/TruncatedRealMutatorGA/')

    y = np.min(costs, axis=1)
    plt.plot(y)
    plt.show()
    # for i in generator:
    #    print(i)
