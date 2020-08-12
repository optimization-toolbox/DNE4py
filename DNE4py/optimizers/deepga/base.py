import numpy as np
import pickle
import random

from abc import ABC, abstractmethod

from DNE4py.utils import MPIData, MPILogger

from DNE4py.optimizers.optimizer import Optimizer
#from .mutation import RealMutator
#from .selection import TruncatedSelection

class BaseGA(Optimizer):

    def __init__(self, objective_function, config):

        super().__init__(objective_function, config)

        # self.initial_guess
        # workers_per_rank
        # num_elite
        # num_parents
        # sigma
        # output_folder
        # global_seed
        # save
        # verbose

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # mutator and selection
        self.mutator_initialize()
        self.selection_initialize()

        # Internal:
        self.generation = 0
        self.population_size = self._size * self.workers_per_rank

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
                self.mpidata_initialguess = MPIData(self.output_folder,
                                                    'initial_guess',
                                                    0)

    #@abstractmethod
    def apply_selection(self, ranks_by_performance):
        pass

    #@abstractmethod
    def apply_mutation(self, ranks_by_performance):
        pass

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

        # Evaluate member:
        self.cost_list = np.zeros(self.workers_per_rank)
        for i in range(len(self.cost_list)):
            self.cost_list[i] = self.objective_function(self.members[i].phenotype)

        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\nPopulation Members (Initial):")
            self.logger.debug(f"| index | seed | x0 | x1 | y |")
            for index in range(len(self.members)):
                seed = self.members[index].genotype
                x0 = self.members[index].phenotype[0].round(10)
                x1 = self.members[index].phenotype[1].round(10)
                y = self.cost_list[index].round(10)
                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} | {y} |")
        # ===================== END LOGGING ===================================

        # Save:
        if (self.save > 0) and (self.generation % self.save == 0):
            self.mpi_save(self.generation)

        # Broadcast fitness:
        cost_matrix = np.empty((self._size, self.workers_per_rank))
        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
                             [cost_matrix, self._MPI.FLOAT])

        # Truncate Selection (broadcast genotypes and update members):
        order = np.argsort(cost_matrix.flatten())
        rank = np.argsort(order)
        ranks_and_members_by_performance = rank.reshape(cost_matrix.shape)

        # Apply selection:
        self.apply_selection(ranks_and_members_by_performance)

        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\nPopulation Members (After Selection):")
            self.logger.debug(f"| index | seed | x0 | x1 |")
            for index in range(len(self.members)):
                seed = self.members[index].genotype
                x0 = self.members[index].phenotype[0].round(10)
                x1 = self.members[index].phenotype[1].round(10)
                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} |")
        # ===================== END LOGGING ===================================

        # Apply mutations:
        self.apply_mutation(ranks_and_members_by_performance)

        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\nPopulation Members (After Mutation):")
            self.logger.debug(f"| index | seed | x0 | x1 |")
            for index in range(len(self.members)):
                seed = self.members[index].genotype
                x0 = self.members[index].phenotype[0].round(10)
                x1 = self.members[index].phenotype[1].round(10)
                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} |")
        # ===================== END LOGGING ===================================

        # Next generation:
        self.generation += 1

    def mpi_save(self, s):

        self.mpidata_genotypes.write(self.genotypes)
        self.mpidata_costs.write(self.costs)
        if s == 0:
            self.mpidata_initialguess.write(self.initial_guess)

    @ property
    def genotypes(self):
        genotypes = []
        for member in self.members:
            genotypes.append(member.genotype)
        return np.array(genotypes, dtype=object)

    @ property
    def costs(self):
        return self.cost_list
