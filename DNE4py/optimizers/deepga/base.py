import numpy as np
import json
import random

from abc import ABC, abstractmethod

from DNE4py.mpi_extensions import MPISaver, MPILogger

#from DNE4py.optimizers.optimizer import Optimizer

class BaseGA:

    r'''
    Example config:

    id: TruncatedRealMutatorGA
    initial_guess = np.array([-0.3, 0.7])
    workers_per_rank: 10
    num_elite: 3
    num_parents: 5
    sigma: 0.05
    global_seed: 42
    output_folder: 'results/DeepGA/TruncatedRealMutatorGA'
    verbose: 0

    Description:

    * output_folder (int)
        => path for the raw_data that will be processed with 
    postprocessing module

    * save_steps (int)
        => save per each save_steps generations

    * verbose (int)
        => 0 no printing
        => 1 print number of generations with rank 0
        => 2 save debug file

    '''

    def __init__(self, config):

        # config.get('env_options').get('max_iteration')

        # super().__init__(**config)

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # Configuration:
        self.id = config.get('id')
        self.workers_per_rank = config.get('workers_per_rank')
        self.num_elite = config.get('num_elite')
        self.num_parents = config.get('num_parents')
        self.sigma = config.get('sigma')
        self.global_seed = config.get('global_seed')

        self.output_folder = config.get('output_folder')
        self.save_steps = config.get('save_steps')
        self.verbose = config.get('verbose')

        self.initial_guess = config.get('initial_guess')

        # Initialize Mutator and Selection
        self.mutator_initialize()
        self.selection_initialize()

        # Internal:
        self.generation = 0
        self.population_size = self._size * self.workers_per_rank

        # Logger and DataCollector for MPI:
        # if self.extra_options.get('output_folder') != None:
        #    self.logger = MPILogger(self.output_folder, 'debug_logger', self._rank)

        if self.output_folder != None:

            self.saver_genotypes = MPISaver(f'{self.output_folder}/genotypes_w{self._rank}.npy')
            self.saver_costs = MPISaver(f'{self.output_folder}/costs_w{self._rank}.npy')
        if self._rank == 0:
            self.saver_initialguess = MPISaver(f'{self.output_folder}/initial_guess.npy')

    #@abstractmethod
    def apply_selection(self, ranks_by_performance):
        pass

    #@abstractmethod
    def apply_mutation(self, ranks_by_performance):
        pass

    def run(self, objective_function, steps):

        self.objective_function = objective_function

        if self._rank == 0:
            self.saver_initialguess.write(self.initial_guess)

        for _ in range(steps):

            # =================== LOGGING =====================================
            if self.verbose == 1:
                if self._rank == 0:
                    print(f"Generation: {self.generation}/{steps}")
            elif self.verbose == 2:
                self.logger.debug(f"\nGeneration: {self.generation}/{steps}")
            # =================== LOGGING =====================================

            self.step()

        # CLOSING:
        if self._rank == 0:
            if self.output_folder != None:
                info = {'nb_generations': steps,
                        'sigma': self.sigma}
                with open(f'{self.output_folder}/info.json', "w") as f:
                    json.dump(info, f)

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

        # ======================= SAVING =====================================
        if (self.generation % self.save_steps == 0):
            self.saver_genotypes.write(self.genotypes)
            self.saver_costs.write(self.costs)
        else:
            self.saver_genotypes.write(np.array(np.nan))
            self.saver_costs.write(np.array(np.nan))
        # ===================== END SAVING ===================================

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

    @ property
    def genotypes(self):
        genotypes = []
        for member in self.members:
            genotypes.append(member.genotype)
        return np.array(genotypes, dtype=object)

    @ property
    def costs(self):
        return self.cost_list
