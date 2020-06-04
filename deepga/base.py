import numpy as np
import pickle
import random
from functools import partial
from .sliceops import *
from abc import ABC, abstractmethod
from collections import Counter

from .logger import MPILogger, MPIData


class BaseGA(ABC):

    def __init__(self, **kwargs):

        # Initiate MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()

        # Logger for MPI:
        self.logger = MPILogger(self._rank)

        # Input:
        self.objective = kwargs.get('objective', None)
        self.initial_guess = kwargs.get('initial_guess').astype(dtype=np.float32)
        self.num_elite = kwargs.get('num_elite', 1)
        self.global_seed = kwargs.get('seed', 1)

        self.members_per_rank = kwargs.get('members_per_rank', 10)
        self.population_size = self._size * self.members_per_rank

        # Internal Parameters:
        self._generation_number = 0

    @abstractmethod
    def apply_selection(self, ranks_by_performance):
        pass

    @abstractmethod
    def apply_mutation(self, ranks_by_performance):
        pass

    def __call__(self, n_steps, save=-1):

        # Prepare save file:
        if save > 0:
            self.info_genealogies = MPIData('genealogy', self._rank)
            self.info_costs = MPIData('costs', self._rank)

        for i in range(n_steps):

            self.logger.debug(f"Generation:  {self._generation_number}")
            self._step()

            if (save > 0) and (i % save == 0):
                self.info_genealogies.write(self.genealogies)
                self.info_costs.write(self.costs)

    def _step(self):

        # Evaluate member:
        self.cost_list = np.zeros(self.members_per_rank, dtype=np.float32)
        for i in range(len(self.cost_list)):
            self.cost_list[i] = self.objective(self.members[i].parameters)

        # Broadcast fitness:
        cost_matrix = np.empty((self._size, self.members_per_rank),
                               dtype=np.float32)
        self._comm.Allgather([self.cost_list, self._MPI.FLOAT],
                             [cost_matrix, self._MPI.FLOAT])

        # Truncate Selection (broadcast genealogies and update members):
        order = np.argsort(cost_matrix.flatten())
        rank = np.argsort(order)
        ranks_and_members_by_performance = rank.reshape(cost_matrix.shape)

        self.apply_selection(ranks_and_members_by_performance)

        # Apply mutations:
        self.apply_mutation(ranks_and_members_by_performance)

        # Next generation_number:
        self._generation_number += 1

    @ classmethod
    def load(cls, filename):
        pass

    @ property
    def best(self):
        """
        :return: generates and returns the best member of the population.
            Only Rank==0 should be accessing this property.
        """
        pass

    @ property
    def genealogies(self):
        """
        :return: a list of all members of the current population.
            Only Rank==0 should be accessing this property.
        """
        genealogies = []
        for member in self.members:
            genealogies.append(np.array(member.genealogy))
        return np.array(genealogies)

    @ property
    def costs(self):
        """
        :return: scores of all members of the current population. In same order
            as population list. Only Rank==0 should be accessing this property.
        """
        return self.cost_list

    def __getstate__(self):

        state = {"num_elite": self.num_elite,
                 "_max_seed": self._max_seed,
                 "global_seed": self.global_seed,
                 "rgn_selection": self.rgn_selection,
                 "rgn_init": self.rgn_init,
                 "seedlist_init_params": self.seedlist_init_params,
                 "seedlist_init_epsilons": self.seedlist_init_epsilons,
                 "_mutation_rng": self._mutation_rng,
                 "rng_epsilons": self.rng_epsilons,
                 "_generation_number": self._generation_number,
                 "_cost_history": self._cost_history,
                 "_member_genealogy": self._member_genealogy,
                 "initial_guess": self.initial_guess}

        return state

    def __setstate__(self, state):
        """
        Member genealogy must be reset to the corresponding rank, since
        only rank0 is saved.
        """
        for key in state:
            setattr(self, key, state[key])

        # Reconstruct larger structures and load MPI
        from mpi4py import MPI
        self._MPI = MPI
        self._comm = MPI.COMM_WORLD
        self._size = self._comm.Get_size()
        self._rank = self._comm.Get_rank()
        self.objective = None
        self.obj_args = None

        # Reassign member genealogies and make seed rng recent
        self.rng_epsilons = np.random.RandomState(self.seedlist_init_epsilons[self._rank])
        for i in range(self._generation_number):
            self.rng_epsilons.randint(0, self._max_seed)
        self._member_genealogy = self._population_genealogy[self._rank][:]
