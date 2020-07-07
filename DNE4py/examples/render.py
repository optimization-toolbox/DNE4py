# After running these methods, you can generate a compressed .gif in the following command:
# $: convert -layers OptimizeTransparency -delay 20 -loop 0 `ls -v` myimage.gif
# To Optimizer: (Use Optimize Transparency 10%) https://ezgif.com/optimize/


import pickle

from scipy.stats import multivariate_normal

from DNE4py.postprocessing.utils import load_mpidata
from DNE4py.optimizers.cmaes import CMAES


def randomsearch_render(folder_path, nb_generations, objective):

    import numpy as np
    import matplotlib.pyplot as plt

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

    for g in range(nb_generations - 1):

        # Image 1:

        # Plot function:
        fig, ax = start_image()
        ax.pcolormesh(x1, x2, Y, cmap=plt.cm.coolwarm)

        # Plot points:
        ax.scatter(genotypes[g][:, 0], genotypes[g][:, 1], s=10, c='black')
        plt.savefig(f"pp_{folder_path}/{g+1}_1.jpeg")

def cmaes_render(folder_path, nb_generations, objective, sigma):

    import numpy as np
    import matplotlib.pyplot as plt

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
    plt.savefig(f"pp_{folder_path}/0.jpeg")

    cmaes = CMAES(objective=objective,
                  initial_guess=initial_guess,
                  workers_per_rank=10,
                  sigma=sigma,
                  seed=100,
                  save=0,
                  verbose=0,
                  output_folder='DNE4py_result')
    optimizer = cmaes.optimizer

    # Loop:
    contour_x0, contour_x1 = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.dstack((contour_x0, contour_x1))
    for g in range(nb_generations - 1):

        # Image 1:

        # Plot function:
        fig, ax = start_image()
        ax.pcolormesh(x1, x2, Y, cmap=plt.cm.coolwarm)

        # Plot current distribution (black):
        mu_x0, mu_x1 = optimizer.mean
        variance = optimizer.sigma
        rv = multivariate_normal([mu_x0, mu_x1], [[variance, 0], [0, variance]])
        ax.contour(contour_x0, contour_x1, rv.pdf(pos), colors='black', alpha=0.3)
        # plt.savefig(f"{folder_path}/pp/{g+1}_1.jpeg")

        # Plot current points (black):
        ax.scatter(genotypes[g][:, 0], genotypes[g][:, 1], s=10, c='black')
        plt.savefig(f"pp_{folder_path}/{g+1}_1.jpeg")

        # Update CMAES
        solutions = np.array(optimizer.ask())
        optimizer.tell(genotypes[g], costs[g])

        # Plot next distribution (red):
        mu_x0, mu_x1 = optimizer.mean
        variance = optimizer.sigma
        rv = multivariate_normal([mu_x0, mu_x1], [[variance, 0], [0, variance]])
        ax.contour(contour_x0, contour_x1, rv.pdf(pos), colors='red', alpha=0.3)
        plt.savefig(f"pp_{folder_path}/{g+1}_2.jpeg")

        # Plot current points (red):
        ax.scatter(genotypes[g + 1][:, 0], genotypes[g + 1][:, 1], s=10, c='red')
        plt.savefig(f"pp_{folder_path}/{g+1}_3.jpeg")

def deepga_render(folder_path, nb_generations, objective, sigma, num_parents, num_elite):

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
    plt.savefig(f"pp_{folder_path}/0.jpeg")

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
        plt.savefig(f"pp_{folder_path}/{g+1}_1.jpeg")

        # Image 2:
        order = np.argsort(costs[g])
        rank = np.argsort(order)
        parents_mask = rank < num_parents
        elite_mask = rank < num_elite

        parents_phenotypes = phenotypes[parents_mask]
        elite_phenotypes = phenotypes[elite_mask]
        ax.scatter(parents_phenotypes[:, 0], parents_phenotypes[:, 1], c='green', s=10)
        ax.scatter(elite_phenotypes[:, 0], elite_phenotypes[:, 1], c='blue', s=10)
        plt.savefig(f"pp_{folder_path}/{g+1}_2.jpeg")

        # Image 3
        phenotypes = []
        for genotype in genotypes[g + 1]:
            phenotype = Member(initial_guess, genotype, sigma).phenotype
            phenotypes.append(phenotype)
        phenotypes = np.array(phenotypes)
        ax.scatter(phenotypes[:, 0], phenotypes[:, 1], c='red', s=10)
        plt.savefig(f"pp_{folder_path}/{g+1}_3.jpeg")
