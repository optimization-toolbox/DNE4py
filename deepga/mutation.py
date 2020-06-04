import random

import numpy as np

from .base import BaseGA

class Member:

    def __init__(self,
                 init_parameters,
                 init_gene,
                 rng_epsilons,
                 sigma):

        # Define initial parameters
        self.initial_parameter = init_parameters
        self.parameters = self.initial_parameter.copy()

        # Init Random Generators (rng):
        self.rng_epsilons = rng_epsilons
        self.rng = np.random.RandomState()

        # Size and Mutation rate
        self.sigma = sigma
        self.size = len(self.parameters)

        # Define genealogy and parameters
        self.new([init_gene])

    # SELECTION
    def new(self, new_genealogy):

        # Set genealogy
        self.genealogy = new_genealogy

        # Set parameters:
        self.parameters[:] = self.initial_parameter[:]
        for seed in self.genealogy:
            self.rng.seed(seed)
            self.parameters += self.rng.randn(self.size) * self.sigma

    # MUTATION
    def mutate(self):

        # New gene
        seed = self.rng_epsilons.randint(0, 2 ** 32 - 1)
        self.genealogy.append(seed)

        # New param
        self.rng.seed(seed)
        self.parameters += self.rng.randn(self.size) * self.sigma


class RealMutator(BaseGA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Input:
        self.sigma = kwargs.get('sigma', 1.0)

        # Init seedmatrix:
        self.rgn_init = random.Random(self.global_seed)

        self.seedmatrix_init_genes = np.zeros((self._size, self.members_per_rank), dtype=np.int64)
        for i in range(self._size):
            row = self.rgn_init.sample(range(2**32 - 1), self.members_per_rank)
            self.seedmatrix_init_genes[i] = row

        self.seedmatrix_init_epsilons = np.zeros((self._size, self.members_per_rank), dtype=np.int64)
        for i in range(self._size):
            row = self.rgn_init.sample(range(2**32 - 1), self.members_per_rank)
            self.seedmatrix_init_epsilons[i] = row

        # Init rng_epsilons_list
        self.rng_epsilons_list = [np.random.RandomState(self.seedmatrix_init_epsilons[self._rank, i]) for i in range(self.members_per_rank)]

        # Init genes:
        self.init_genes = self.seedmatrix_init_genes[self._rank]

        self.members = [Member(init_parameters=self.initial_guess,
                               init_gene=self.init_genes[i],
                               rng_epsilons=self.rng_epsilons_list[i],
                               sigma=self.sigma)
                        for i in range(self.members_per_rank)]

    def apply_mutation(self, ranks_and_members_by_performance):
        """
        Preserve the top num_elite members, mutate the rest
        """
        elite_mask = ranks_and_members_by_performance < self.num_elite
        row, column = np.where(elite_mask)
        elite_tuples = tuple(zip(row, column))

        for rank, member_id in elite_tuples:
            if rank == self._rank:
                self.members[member_id].mutate()

        # for i, rank in enumerate(ranks_and_members_by_performance[self.num_elite:]):
        #    if rank == self._rank:
        #        self.member.mutate()

    def __getstate__(self):
        state = super().__getstate__()
        state["sigma"] = self.sigma
        state["_member"] = self._member

        return state

    def __setstate__(self, state):
        """
        The members need to be generated from the member's genealogy,
        as only the rank=0 member is actually saved.
        The member genealogy is set to the proper rank by the baseGA
        """
        super().__setstate__(state)
        self._member = self._make_member(self._mutation_rng,
                                         self._member_genealogy)
