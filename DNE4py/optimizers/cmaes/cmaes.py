import numpy as np
#import pickle
#import random

#from abc import ABC, abstractmethod

from DNE4py.utils import MPIData, MPILogger
import cma


class CMAES:

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
        self.sigma = kwargs.get('sigma')
        self.global_seed = kwargs.get('seed')

        self.save = kwargs.get('save', 0)
        self.verbose = kwargs.get('verbose', 0)
        self.output_folder = kwargs.get('output_folder', 'DNE4py_result')

        self.workers_per_rank = kwargs.get('workers_per_rank')
        self.population_size = self._size * self.workers_per_rank

        # Internal:
        self.optimizer = cma.CMAEvolutionStrategy(self.initial_guess, self.sigma, {'verb_disp': 0, 'seed': self.global_seed, 'popsize': self.population_size})
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
                self.mpidata_initialguess = MPIData(self.output_folder,
                                                    'initial_guess',
                                                    0)

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

        # Ask solutions:
        self.solutions = np.array(self.optimizer.ask())

        # Evaluate only your own solutions:
        self.my_solutions = self.solutions[self.my_ids]
        self.cost_list = np.zeros(self.workers_per_rank)
        for i in range(len(self.cost_list)):
            self.cost_list[i] = self.objective(self.my_solutions[i])

        # Broadcast fitness:
        self.all_costs = np.empty(self.population_size)
        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
                             [self.all_costs, self._MPI.FLOAT])

        # Tell solutions and evaluations:
        self.optimizer.tell(self.solutions, self.all_costs)

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

        # Evaluate member:
    #        self.cost_list = np.zeros(self.workers_per_rank)
#        for i in range(len(self.cost_list)):
#            self.cost_list[i] = self.objective(self.members[i].phenotype)


#        # Save:
#        if (self.save > 0) and (self.generation % self.save == 0):
#            self.mpi_save(self.generation)

#        # Broadcast fitness:
#        cost_matrix = np.empty((self._size, self.workers_per_rank))
#        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
#                             [cost_matrix, self._MPI.FLOAT])

#        # Truncate Selection (broadcast genotypes and update members):
#        order = np.argsort(cost_matrix.flatten())
#        rank = np.argsort(order)
#        ranks_and_members_by_performance = rank.reshape(cost_matrix.shape)

        # Apply selection:
#        self.apply_selection(ranks_and_members_by_performance)

#        # ======================= LOGGING =====================================
#        if self.verbose == 2:
#            self.logger.debug(f"\nPopulation Members (After Selection):")
#            self.logger.debug(f"| index | seed | x0 | x1 |")
#            for index in range(len(self.members)):
#                seed = self.members[index].genotype
#                x0 = self.members[index].phenotype[0].round(10)
#                x1 = self.members[index].phenotype[1].round(10)
#                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} |")
        # ===================== END LOGGING ===================================

        # Apply mutations:
#        self.apply_mutation(ranks_and_members_by_performance)

#        # ======================= LOGGING =====================================
#        if self.verbose == 2:
#            self.logger.debug(f"\nPopulation Members (After Mutation):")
#            self.logger.debug(f"| index | seed | x0 | x1 |")
#            for index in range(len(self.members)):
#                seed = self.members[index].genotype
#                x0 = self.members[index].phenotype[0].round(10)
#                x1 = self.members[index].phenotype[1].round(10)
#                self.logger.debug(f"| {index} | {seed} | {x0} | {x1} |")
#        # ===================== END LOGGING ===================================

#        # Next generation:
#        self.generation += 1

    def mpi_save(self, s):

        self.mpidata_genotypes.write(self.my_solutions)
        self.mpidata_costs.write(self.cost_list)
        if s == 0:
            self.mpidata_initialguess.write(self.initial_guess)
