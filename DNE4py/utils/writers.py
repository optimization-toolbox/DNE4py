import os
import pickle
import logging

import numpy as np


class MPIWriter:

    def __init__(self, folder_path, name, rank):

        self.rank = rank
        self.folder_path = folder_path
        self.name = name

        self.filename = f'{folder_path}/{name}_w{rank}'

        assert os.path.isdir(folder_path) == True, f'You should create the folder: {folder_path}'
        os.path.isfile(self.filename)


class MPIData(MPIWriter):

    def __init__(self, folder_path, name, rank):
        super().__init__(folder_path, name, rank)
        self.filename += '.pkl'

        assert os.path.isfile(self.filename) == False, f'You should delete all files inside the folder: {folder_path}'

    def write(self, data):
        with open(self.filename, 'ab') as f:
            pickle.dump(data, f)


class MPILogger(MPIWriter):

    def __init__(self, folder_path, name, rank):
        super().__init__(folder_path, name, rank)
        self.filename += '.log'

        assert os.path.isfile(self.filename) == False, f'You should delete all files inside the folder: {folder_path}'

        logging.basicConfig(filename=self.filename,
                            level=logging.DEBUG,
                            format='%(message)s')

    def debug(self, msg):
        logging.debug(msg)
