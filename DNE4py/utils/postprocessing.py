import os
import glob
import logging
import numpy as np
import pickle

def load_mpidata(folder_path, name, nb_generations):

    filename = f'{folder_path}/{name}'

    nb_files = len(glob.glob1(f'{folder_path}', f'{name}*'))

    full_data = [[]] * nb_generations
    for w in range(nb_files):
        with open(f'{filename}_w{w}.pkl', 'rb') as f:
            for g in range(nb_generations):
                worker_data = pickle.load(f).tolist()
                full_data[g] = full_data[g] + worker_data
    full_data = np.array(full_data)
    return full_data

def get_best_phenotype(folder_path, nb_generations, sigma):

    from DNE4py.optimizers.deepga.mutation import Member

    # Read Input:
    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)
    genotypes = load_mpidata(f"{folder_path}", "genotypes", nb_generations)
    initial_guess = load_mpidata(f"{folder_path}", "initial_guess", 1)[0]

    # Select Best Idxs:
    best_idxs = np.unravel_index(costs.argmin(), costs.shape)

    # Create member and get phonetype:
    phenotype = Member(initial_guess, genotypes[best_idxs], sigma).phenotype
    return phenotype

def print_statistics(folder_path, nb_generations):

    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)

    final_best_cost = np.min(costs[-1])
    best_cost = np.min(np.min(costs, axis=1))

    print(f"Final Best cost: {final_best_cost}")
    print(f"Best cost: {best_cost}")


def plot_cost_over_generation(folder_path, nb_generations):

    import matplotlib.pyplot as plt

    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)

    means = np.mean(costs, axis=1)
    stds = np.std(costs, axis=1)
    mins = np.min(costs, axis=1)
    maxes = np.max(costs, axis=1)

    plt.errorbar(np.arange(nb_generations), means, stds, fmt='ok', lw=3)
    plt.errorbar(np.arange(nb_generations), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='gray', lw=1)
    plt.xlim(-1, nb_generations)
    plt.xticks(np.arange(-1, nb_generations + 1, 1.0))
    plt.show()

def plot_best_cost_over_generation(folder_path, nb_generations):

    import matplotlib.pyplot as plt

    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)

    mins = np.min(costs, axis=1)
    plt.plot(np.arange(nb_generations), mins)
    plt.xlim(-1, nb_generations)
    plt.xticks(np.arange(-1, nb_generations + 1, 1.0))
    plt.show()

def render_population_over_generation(folder_path, nb_generations, objective, sigma, num_parents, num_elite):

    import numpy as np
    import matplotlib.pyplot as plt

    from DNE4py.optimizers.deepga.mutation import Member

    def start_image():
        fig, ax = plt.subplots()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_xticks(np.arange(-1, 1, 0.2))
        ax.set_yticks(np.arange(-1, 1, 0.2))
        return fig, ax

    # Read Input:
    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)
    genotypes = load_mpidata(f"{folder_path}", "genotypes", nb_generations)
    initial_guess = load_mpidata(f"{folder_path}", "initial_guess", 1)[0]

    # Start figure:
    fig, ax = start_image()

    # Plot function:
    resolution = 100
    x1, x2 = np.meshgrid(np.linspace(-1, 1, resolution), np.linspace(-1, 1, resolution))
    X = np.array([x1.flatten(), x2.flatten()]).T
    Y = np.array([objective(x) for x in X]).reshape(resolution, resolution)
    ax.pcolormesh(x1, x2, Y, cmap=plt.cm.coolwarm)

    # Show Initial Guess
    ax.scatter(initial_guess[0], initial_guess[1], c='black', s=20)
    plt.savefig(f"{folder_path}/pp/0.png")

    # Loop:
    for g in range(nb_generations - 1):

        # Plot function:
        fig, ax = start_image()
        ax.pcolormesh(x1, x2, Y, cmap=plt.cm.coolwarm)

        # Image 1:
        phenotypes = []
        for genotype in genotypes[g]:
            phenotype = Member(initial_guess, genotype, sigma).phenotype
            phenotypes.append(phenotype)
        phenotypes = np.array(phenotypes)

        ax.scatter(phenotypes[:, 0], phenotypes[:, 1], c='black', s=10)
        plt.savefig(f"{folder_path}/pp/{g+1}_1.png")

        # Image 2:
        order = np.argsort(costs[g])
        rank = np.argsort(order)
        parents_mask = rank < num_parents
        elite_mask = rank < num_elite

        parents_phenotypes = phenotypes[parents_mask]
        elite_phenotypes = phenotypes[elite_mask]
        ax.scatter(parents_phenotypes[:, 0], parents_phenotypes[:, 1], c='green', s=10)
        ax.scatter(elite_phenotypes[:, 0], elite_phenotypes[:, 1], c='blue', s=10)
        plt.savefig(f"{folder_path}/pp/{g+1}_2.png")

        # Image 3
        phenotypes = []
        for genotype in genotypes[g + 1]:
            phenotype = Member(initial_guess, genotype, sigma).phenotype
            phenotypes.append(phenotype)
        phenotypes = np.array(phenotypes)
        ax.scatter(phenotypes[:, 0], phenotypes[:, 1], c='red', s=10)
        plt.savefig(f"{folder_path}/pp/{g+1}_3.png")
