import os

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np

import pickle

class MPIData:

    def __init__(self, name, worker):
        self.filename = f'results/experiment/{name}_w{worker}.pkl'
        assert os.path.isdir('results/experiment') == True, 'You should create the folder results/experiment'
        assert os.path.isfile(self.filename) == False, 'You should delete all files inside the folder results/experiment'

    def write(self, data):
        print(data)
        with open(self.filename, 'ab') as f:
            pickle.dump(data, f)


class MPILogger:
    def __init__(self, my_rank):
        self.my_rank = my_rank

    def debug(self, msg, rank=0):
        if self.my_rank == rank:
            logging.debug(msg)
