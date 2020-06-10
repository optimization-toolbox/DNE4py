import os

import logging

import numpy as np

import pickle

def join_mpidata(name, nb_workers, nb_generations):

    path = f'results/experiment/{name}'

    full_data = [[]] * nb_generations
    for w in range(nb_workers):
        with open(f'{path}_w{w}.pkl', 'rb') as f:
            for g in range(nb_generations):
                worker_data = pickle.load(f).tolist()
                full_data[g] = full_data[g] + worker_data
    full_data = np.array(full_data)

    # Save:
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(full_data, f)

def load_mpidata(name):

    path = f'results/experiment/{name}'

    with open(f'{path}.pkl', 'rb') as f:
        data = pickle.load(f)

    return data


class MPIData:

    def __init__(self, name, worker):

        logging.basicConfig(level=logging.DEBUG)

        self.filename = f'results/experiment/{name}_w{worker}.pkl'
        assert os.path.isdir('results/experiment') == True, 'You should create the folder results/experiment'
        assert os.path.isfile(self.filename) == False, 'You should delete all files inside the folder results/experiment'

    def write(self, data):
        with open(self.filename, 'ab') as f:
            pickle.dump(data, f)


class MPILogger:
    def __init__(self, my_rank):
        self.my_rank = my_rank

    def debug(self, msg, rank=0):
        if self.my_rank == rank:
            logging.debug(msg)
