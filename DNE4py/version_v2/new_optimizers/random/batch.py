import numpy as np
#import pickle
#import random

#from abc import ABC, abstractmethod

from ..optimizer import Optimizer
from DNE4py.utils.mpi_extensions import MPISaver, MPILogger

class BatchRandomSearch(Optimizer):

    def __init__(self, objective_function, config):

        super().__init__(objective_function, config)

        # self.objective_function
        # self.dim
        # self.workers_per_rank
        # self.bounds
        # self.global_seed
        # self.save
        # self.verbose
        # self.output_folder

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # Input:
        #self.objective_function = kwargs.get('objective_function')
        #self.initial_guess = kwargs.get('initial_guess')
        #self.bounds = kwargs.get('bounds')
        #self.global_seed = kwargs.get('seed')

        #self.save = kwargs.get('save', 0)
        #self.verbose = kwargs.get('verbose', 0)
        #self.output_folder = kwargs.get('output_folder', 'DNE4py_result')

        #self.workers_per_rank = kwargs.get('workers_per_rank')

        # Internal:
        self.population_size = self._size * self.workers_per_rank
        self.generator = np.random.RandomState(self.global_seed)
        self.my_ids = np.array_split(range(self.population_size), self._size)[self._rank]
        self.generation = 0

        # Logger and DataCollector for MPI:
        if (self.verbose == 2) or (self.save > 0):

            if self.verbose == 2:
                self.logger = MPILogger(self.output_folder, 'debug_logger', self._rank)
            if self.save > 0:
                self.mpidata_genotypes = MPIData(self.output_folder,
                                                 'genotypes',
                                                 self._rank)
                self.mpidata_costs = MPIData(self.output_folder,
                                             'costs',
                                             self._rank)

    def run(self, steps):

        for _ in range(steps):

            # =================== LOGGING =====================================
            if self.verbose > 0 and self._rank == 0:
                print(f"Generation: {self.generation}/{steps}")
            elif self.verbose == 2:
                self.logger.debug(f"\nGeneration: {self.generation}/{steps}")
            # =================== LOGGING =====================================

            self.step()

    def step(self):

        # Get solutions:
        self.solutions = self.generator.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))

        # Evaluate only your own solutions:
        self.my_solutions = self.solutions[self.my_ids]
        self.cost_list = np.zeros(self.workers_per_rank)
        for i in range(len(self.cost_list)):
            self.cost_list[i] = self.objective_function(self.my_solutions[i])

        # Save:
        if (self.save > 0) and (self.generation % self.save == 0):
            self.mpi_save(self.generation)

        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\n| index | x0 | x1 | y |")
            for index, sol in enumerate(self.solutions):

                x0 = sol[0].round(10)
                x1 = sol[1].round(10)
                y = self.all_costs[index].round(10)
                if index in self.my_ids:
                    self.logger.debug(f"| {index} | {x0} | {x1} | {y} | X |")
                else:
                    self.logger.debug(f"| {index} | {x0} | {x1} | {y} |")
        # ===================== END LOGGING ===================================

        # Next generation:
        self.generation += 1

    def mpi_save(self, s):

        self.mpidata_genotypes.write(self.my_solutions)
        self.mpidata_costs.write(self.cost_list)
