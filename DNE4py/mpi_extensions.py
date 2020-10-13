import os
import pickle
import logging

import numpy as np

class MPISaver:

    def __init__(self, file_path):

        self.file_path = file_path
        self.folder_path, self.filename = os.path.split(self.file_path)

        try:
            os.makedirs(self.folder_path)
        except:
            pass

        assert os.path.isfile(self.file_path) == False, f'You should delete all files inside the folder: {self.folder_path} | {self.file_path}'

    def write(self, data):
        with open(self.file_path, 'ab') as f:
            np.save(f, data)

class MPILogger:

    def __init__(self, file_path):

        self.file_path = file_path
        self.folder_path, self.filename = os.path.split(file_path)

        try:
            os.makedirs(self.folder_path)
        except:
            pass

        assert os.path.isfile(self.file_path) == False, f'You should delete all files inside the folder: {self.folder_path} | {self.file_path}'

        logging.basicConfig(filename=self.file_path,
                            level=logging.DEBUG,
                            format='%(message)s')

    def debug(self, msg):
        logging.debug(msg)
