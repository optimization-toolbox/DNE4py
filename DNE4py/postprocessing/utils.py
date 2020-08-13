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
    if name == 'genotypes':
        full_data = np.array(full_data, dtype=object)
    else:
        full_data = np.array(full_data)
    return full_data

def get_best_phenotype(folder_path, nb_generations, sigma):

    from DNE4py.optimizers.deepga.mutation import Member

    # Read Input:
    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)
    genotypes = load_mpidata(f"{folder_path}", "genotypes", nb_generations)
    initial_guess = load_mpidata(f"{folder_path}", "initial_guess", 1)[0]

    best_idxs = np.unravel_index(costs.argmin(), costs.shape)

    # Create member and get phonetype:
    phenotype = Member(initial_guess, genotypes[best_idxs], sigma).phenotype
    return phenotype

def get_best_phenotype_composite(folder_path, nb_generations, sigma):

    from DNE4py.optimizers.deepga.mutation import Member

    # Read Input:
    rankings = load_mpidata(f"{folder_path}", "rankings", nb_generations) 
    genotypes = load_mpidata(f"{folder_path}", "genotypes", nb_generations)
    initial_guess = load_mpidata(f"{folder_path}", "initial_guess", 1)[0]

    best_idxs = np.unravel_index(rankings.argmin(), rankings.shape)

    # Create member and get phonetype:
    phenotype = Member(initial_guess, genotypes[best_idxs], sigma).phenotype
    return phenotype

    # # Read Input:
    # if loss_type == "compact":
    #     costs = load_mpidata(f"{folder_path}", "costs", nb_generations)
    # elif loss_type == "composite":
    #     costs = load_mpidata(f"{folder_path}", "costs", nb_generations)
def get_best_phenotype_generator(folder_path, nb_generations, sigma):

    from DNE4py.optimizers.deepga.mutation import Member

    # Read Input:
    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)
    genotypes = load_mpidata(f"{folder_path}", "genotypes", nb_generations)
    initial_guess = load_mpidata(f"{folder_path}", "initial_guess", 1)[0]

    # Select Best Idxs:
    min_idxs = np.argmin(costs, axis=1)
    for i in range(nb_generations):
        genotype = genotypes[i, min_idxs[i]]
        yield Member(initial_guess, genotype, sigma).phenotype

    
def get_best_phenotype_generator_composite(folder_path, nb_generations, sigma):

    from DNE4py.optimizers.deepga.mutation import Member

    # Read Input:
    rankings = load_mpidata(f"{folder_path}", "rankings", nb_generations) 
    genotypes = load_mpidata(f"{folder_path}", "genotypes", nb_generations)
    initial_guess = load_mpidata(f"{folder_path}", "initial_guess", 1)[0]

    # Select Best Idxs:
    min_idxs = np.argmin(rankings, axis=1)
    for i in range(nb_generations):
        genotype = genotypes[i, min_idxs[i]]
        yield Member(initial_guess, genotype, sigma).phenotype


def print_statistics(folder_path, nb_generations):

    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)

    final_best_cost = np.min(costs[-1])
    best_cost = np.min(np.min(costs, axis=1))

    print(f"Final Best cost: {final_best_cost}")
    print(f"Best cost: {best_cost}")

def plot_cost_over_generation(folder_path, nb_generations, sigma=None, test_objective=None):

    import matplotlib.pyplot as plt

    # Load data:
    costs = load_mpidata(f"{folder_path}", "costs", nb_generations)

    # Calculate data:
    means = np.mean(costs, axis=1)
    stds = np.std(costs, axis=1)
    mins = np.min(costs, axis=1)
    maxes = np.max(costs, axis=1)

    # Plot Population errorbar:
    plt.errorbar(np.arange(nb_generations), means, stds, fmt='ok', lw=3)
    plt.errorbar(np.arange(nb_generations), means, [means - mins, maxes - means],
                 fmt='.k', ecolor='gray', lw=1)

    # Plot Best Individuals (blue line):
    plt.plot(np.arange(nb_generations), mins)

    # Plot Test Performance of Best Individuals (red line):
    if test_objective is not None:
        best_phenotypes = get_best_phenotype_generator(folder_path, nb_generations, sigma)

        test_evaluations = []
        for i, phenotype in enumerate(best_phenotypes):
            if i % 4 == 0:
                print(f'{i}/{nb_generations}\r', end='')
                evaluation = test_objective(phenotype)
                test_evaluations.append(evaluation)
        test_evaluations = np.array(test_evaluations)
        plt.plot(np.arange(0, nb_generations, 4), test_evaluations)

    # Configuration:
    plt.xlim(-1, nb_generations)
    plt.xticks(np.arange(-1, nb_generations + 1, nb_generations // 10.0))

    # Save
    plt.savefig(f"{folder_path}/cost_over_generation.png")


def save_meta_losses(input_path, output_path, nb_generations, test_objective=None, sigma=None):

    # Graph 1 (generation x (meta_train_loss, meta_test_loss))

    # Meta-Train Loss:
    # Load data:
    costs = load_mpidata(f"{input_path}", "costs", nb_generations)

    # Calculate data:
    meta_train_loss_y = np.min(costs, axis=1)

    # Meta-Test Loss:
    if test_objective is not None:
        best_phenotypes = get_best_phenotype_generator(input_path, nb_generations, sigma)

        meta_test_loss_y = []
        for i, phenotype in enumerate(best_phenotypes):
            if i % 1 == 0:
                print(f'{i}/{nb_generations}\r', end='')
                evaluation = test_objective(phenotype)
                meta_test_loss_y.append(evaluation)
        meta_test_loss_y = np.array(meta_test_loss_y)

    with open(f"{output_path}/meta_train_loss_y.npy", "wb") as f:
        np.save(f, meta_train_loss_y)

    with open(f"{output_path}/meta_test_loss_y.npy", "wb") as f:
        np.save(f, meta_test_loss_y)
