import numpy as np
import pickle
import random
from functools import partial
from .sliceops import *
from abc import ABC, abstractmethod
from collections import Counter

from neuroevolution4py.utils import MPILogger, MPIData

import time

class BaseGA(ABC):

    def __init__(self, **kwargs):

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # Input:
        self.objective = kwargs.get('objective')
        self.initial_guess = kwargs.get('initial_guess')
        self.num_elite = kwargs.get('num_elite')
        self.global_seed = kwargs.get('seed')

        self.verbose = kwargs.get('verbose', 0)
        self.save = kwargs.get('save', -1)

        self.workers_per_rank = kwargs.get('workers_per_rank')
        self.population_size = self._size * self.workers_per_rank

        # Internal Parameters:
        self._generation_number = 0

        # Logger and DataCollector for MPI:
        if self.verbose == 2:
            self.logger = MPILogger(self._rank)
        if self.save > 0:
            self.mpidata_genealogies = MPIData('genotypes', self._rank)
            self.mpidata_costs = MPIData('costs', self._rank)
            self.mpidata_initialguess = MPIData('initial_guess', 0)

    @abstractmethod
    def apply_selection(self, ranks_by_performance):
        pass

    @abstractmethod
    def apply_mutation(self, ranks_by_performance):
        pass

    def __call__(self, n_steps):

        for s in range(n_steps):
            self.logger.debug(f"\nGeneration: {self._generation_number}")
            self._step(s)

    def _step(self, s):

        # Evaluate member:
        t = time.time()
        self.cost_list = np.zeros(self.workers_per_rank)
        for i in range(len(self.cost_list)):
            self.cost_list[i] = self.objective(self.members[i].phenotype)

        # =============== DEBUG =============================
        self.logger.debug(f"\nPopulation Members (Initial):")
        self.logger.debug(f"| index | seed | x0 | x1 | y |")
        for index in range(len(self.members)):
            seed = self.members[index].genotype
            x0 = self.members[index].phenotype[0].round(10)
            x1 = self.members[index].phenotype[1].round(10)
            y = self.cost_list[index].round(10)
            self.logger.debug(f"| {index} | {seed} | {x0} | {x1} | {y} |")
        # =============== END =============================

        # Save:
        if (self.save > 0) and (s % self.save == 0):
            self.mpi_save(s)

        # Broadcast fitness:
        t = time.time()
        cost_matrix = np.empty((self._size, self.workers_per_rank))
        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
                             [cost_matrix, self._MPI.FLOAT])

        # Truncate Selection (broadcast genealogies and update members):
        order = np.argsort(cost_matrix.flatten())
        rank = np.argsort(order)
        ranks_and_members_by_performance = rank.reshape(cost_matrix.shape)

        # Apply selection:
        self.apply_selection(ranks_and_members_by_performance)

        # =============== DEBUG =============================
        self.logger.debug(f"\nPopulation Members (After Selection):")
        self.logger.debug(f"| index | seed | x0 | x1 |")
        for index in range(len(self.members)):
            seed = self.members[index].genotype
            x0 = self.members[index].phenotype[0].round(10)
            x1 = self.members[index].phenotype[1].round(10)
            self.logger.debug(f"| {index} | {seed} | {x0} | {x1} |")
        # =============== END =============================

        # Apply mutations:
        self.apply_mutation(ranks_and_members_by_performance)

        # =============== DEBUG =============================
        self.logger.debug(f"\nPopulation Members (After Mutation):")
        self.logger.debug(f"| index | seed | x0 | x1 |")
        for index in range(len(self.members)):
            seed = self.members[index].genotype
            x0 = self.members[index].phenotype[0].round(10)
            x1 = self.members[index].phenotype[1].round(10)
            self.logger.debug(f"| {index} | {seed} | {x0} | {x1} |")
        # =============== END =============================

        # Next generation_number:
        self._generation_number += 1

    def mpi_save(self, s):

        self.mpidata_genealogies.write(self.genealogies)
        self.mpidata_costs.write(self.costs)
        if s == 0:
            self.mpidata_initialguess.write(self.initial_guess)

    @ property
    def genealogies(self):
        genealogies = []
        for member in self.members:
            genealogies.append(member.genotype)
        return np.array(genealogies)

    @ property
    def costs(self):
        return self.cost_list
