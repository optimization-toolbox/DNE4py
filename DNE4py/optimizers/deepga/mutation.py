import random

import numpy as np

from .base import BaseGA
from .member import Member

class RealMutator(BaseGA):

    def mutator_initialize(self):

        # Global Random Generator:
        self.rgn_global = random.Random(self.global_seed)

        # Global seedmatrix:
        self.seedmatrix = np.zeros((self._size, self.workers_per_rank), dtype=np.int64)
        for i in range(self._size):
            row = self.rgn_global.sample(range(2**32 - 1), self.workers_per_rank)
            self.seedmatrix[i] = row

        # Generate Genes Random Generator:
        self.rng_genes_list = [np.random.RandomState(self.seedmatrix[self._rank, i]) for i in range(self.workers_per_rank)]

        # Generate Initial genes
        self.initial_genotypes = []
        for i in range(self.workers_per_rank):
            initial_genotype = self.rng_genes_list[i].randint(2**32 - 1)
            self.initial_genotypes.append([[initial_genotype, self.sigma]])

        self.members = [Member(initial_phenotype=self.initial_guess,
                               genotype=self.initial_genotypes[i])
                        for i in range(self.workers_per_rank)]

    def apply_mutation(self, ranks_and_members_by_performance):
        """
        Preserve the top num_elite members, mutate the rest
        """
        no_elite_mask = ranks_and_members_by_performance >= self.num_elite

        row, column = np.where(no_elite_mask)
        no_elite_tuples = tuple(zip(row, column))

        # ======================= LOGGING =====================================
        if self.verbose == 2:
            self.logger.debug(f"\nSelected Elite:")
            self.logger.debug(f"| rank | member_id |")
            elite_mask = np.invert(no_elite_mask)
            row, column = np.where(elite_mask)
            elite_indexes = tuple(zip(row, column))
            for rank, member_id in elite_indexes:
                self.logger.debug(f"| {rank} | {member_id} |")
        # ===================== END LOGGING ===================================

        for rank, member_id in no_elite_tuples:
            if rank == self._rank:
                rng_genes = self.rng_genes_list[member_id]
                self.members[member_id].mutate(rng_genes, self.sigma)
