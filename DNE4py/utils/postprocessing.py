import os
import logging
import numpy as np
import pickle

def join_mpidata(name, nb_workers, nb_generations):

    path = 'results/experiment/' + str(name)

    full_data = [[]] * nb_generations
    for w in range(nb_workers):
        with open(str(path) + '_w' + str(w) + '.pkl', 'rb') as f:
            for g in range(nb_generations):
                worker_data = pickle.load(f).tolist()
                full_data[g] = full_data[g] + worker_data
    full_data = np.array(full_data)

    # Save:
    with open(str(path) + '.pkl', 'wb') as f:
        pickle.dump(full_data, f)

def load_mpidata(name):

    path = 'results/experiment/' + str(name)

    with open(str(path) + '.pkl', 'rb') as f:
        data = pickle.load(f)

    return data
