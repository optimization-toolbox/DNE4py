import random

import numpy as np

from .base import BaseGA


class Member:
    r'''
        Initialization:
            initial_phenotype: initial list of parameters
            initial_gene: initial seed
            sigma: parameter to apply on mutations

        Internal attributes:
            genotype: list of seeds
            phenotype: list of parameters

        Methods:
            recreate(new_genotype):
                update genotype and phenotype from new_genotype
            mutate(rng_genes):
                create a new gene and update genotype and phenotype
    '''

    def __init__(self,
                 initial_phenotype,
                 initial_genotype,
                 sigma):

        # Define initial phenotype:
        self.initial_phenotype = initial_phenotype
        self.phenotype = self.initial_phenotype.copy()

        #self.initial_genotype = initial_genotype.copy()

        # Internal attributes:
        self.rng = np.random.RandomState()
        self.size, self.sigma = len(self.phenotype), sigma

        # Define genotype and phenotype:
        self.recreate(initial_genotype)

    def mutate(self, rng_genes):
        r'''create a new gene and update genotype and phenotype'''

        # Increase genotype:
        seed = rng_genes.randint(0, 2 ** 32 - 1)
        self.genotype.append(seed)

        # Mutate phenotype:
        self.rng.seed(seed)
        self.phenotype += self.rng.randn(self.size) * self.sigma

    def recreate(self, new_genotype):
        r'''update genotype and phenotype from new_genotype'''

        # Set genotype:
        self.genotype = new_genotype[:]

        # Set phenotype:
        self.phenotype[:] = self.initial_phenotype[:]

        for seed in self.genotype:
            self.rng.seed(seed)
            self.phenotype += self.rng.randn(self.size) * self.sigma

class RealMutator(BaseGA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Input:
        self.sigma = kwargs.get('sigma')

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
            self.initial_genotypes.append([initial_genotype])

        self.members = [Member(initial_phenotype=self.initial_guess,
                               initial_genotype=self.initial_genotypes[i],
                               sigma=self.sigma)
                        for i in range(self.workers_per_rank)]

    def apply_mutation(self, ranks_and_members_by_performance):
        """
        Preserve the top num_elite members, mutate the rest
        """
        no_elite_mask = ranks_and_members_by_performance >= self.num_elite

        row, column = np.where(no_elite_mask)
        no_elite_tuples = tuple(zip(row, column))

        # =============== DEBUG =============================
        self.logger.debug(f"\nSelected Elite:")
        self.logger.debug(f"| rank | member_id |")
        elite_mask = np.invert(no_elite_mask)
        row, column = np.where(elite_mask)
        elite_indexes = tuple(zip(row, column))
        for rank, member_id in elite_indexes:
            self.logger.debug(f"| {rank} | {member_id} |")
        # =============== END =============================

        for rank, member_id in no_elite_tuples:
            if rank == self._rank:
                rng_genes = self.rng_genes_list[member_id]
                self.members[member_id].mutate(rng_genes)
