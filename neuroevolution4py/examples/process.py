
import numpy as np
from neuroevolution4py.utils import join_mpidata, load_mpidata
from neuroevolution4py.optimizers.deepga.mutation import Member


initial_guess = np.array([1.0, 1.0])
sigma = 0.1

def objective(x):

    result = np.sum(x**2)

    x1 = x[0].round(10)
    x2 = x[1].round(10)
    y = result.round(10)
    print(f'function: x = [{x1},{x2}], f = {y}')

    return result


# Load weights:
join_mpidata("costs", nb_workers=2, nb_generations=1)
join_mpidata("genotypes", nb_workers=2, nb_generations=1)
costs = load_mpidata("costs")
genotypes = load_mpidata("genotypes")

# print(costs)
# print(genotypes)

for g in range(len(genotypes)):

    print(f"Generation: {g}")
    population_genotype = genotypes[g]

    for genotype in population_genotype:
        member = Member(initial_guess, genotype, sigma)
        phenotype = member.phenotype

        x1 = phenotype[0].round(10)
        x2 = phenotype[1].round(10)
        y = objective(phenotype).round(10)
        print(f'load: [{x1},{x2}], {y}')


# Plot
#means = np.mean(costs, axis=1)
#stds = np.std(costs, axis=1)
#mins = np.min(costs, axis=1)
#maxes = np.max(costs, axis=1)
