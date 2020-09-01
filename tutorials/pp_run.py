import sys
import numpy as np

from DNE4py import load_mpidata, get_best_phenotype, get_best_phenotype_generator


def objective_function(x):
    return np.sum(x**2)


if "__main__" == __name__:

    costs = load_mpidata('costs', 'results/TruncatedRealMutatorGA/')
    genotypes = load_mpidata('genotypes', 'results/TruncatedRealMutatorGA/')
    initial_guess = load_mpidata('initial_guess', 'results/TruncatedRealMutatorGA/')

    x = get_best_phenotype('results/TruncatedRealMutatorGA/')
    print(x)

    generator = get_best_phenotype_generator('results/TruncatedRealMutatorGA/')

    for i in generator:
        print(i)
