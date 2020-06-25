import os
import logging
#logging.basicConfig(level=logging.DEBUG)
import numpy as np
import pickle


class MPIData:

    def __init__(self, name, worker):

        self.filename = 'results/experiment/' + str(name) + '_w' + str(worker) + '.pkl'
        assert os.path.isdir('results/experiment') == True, 'You should create the folder results/experiment'
        assert os.path.isfile(self.filename) == False, 'You should delete all files inside the folder results/experiment'

    def write(self, data):
        with open(self.filename, 'ab') as f:
            pickle.dump(data, f)


class MPILogger:
    def __init__(self, worker):

        self.worker = worker

        self.filename = f'results/experiment/logfile_{self.worker}.log'
        assert os.path.isdir('results/experiment') == True, 'You should create the folder results/experiment'
        assert os.path.isfile(self.filename) == False, 'You should delete all files inside the folder results/experiment'

        logging.basicConfig(filename=self.filename, level=logging.DEBUG, format='%(message)s')

    def debug(self, msg):
        logging.debug(msg)
