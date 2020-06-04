import os

import logging
logging.basicConfig(level=logging.DEBUG)

import numpy as np

class MPIData:

    def __init__(self, name, worker):
        
        self.filename = f'results/experiment/{name}_w{worker}.npy'

    def write(self, arr):
        with open(self.filename, 'wb') as f:
            np.save(f, arr)

class MPILogger:
    def __init__(self, my_rank):
        self.my_rank = my_rank

    def debug(self, msg, rank=0):
        if self.my_rank == rank:
            logging.debug(msg)
