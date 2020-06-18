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
        self.genotype = new_genotype

        # Set phenotype:
        self.phenotype[:] = self.initial_phenotype[:]
        for seed in self.genotype:
            self.rng.seed(seed)
            self.phenotype += self.rng.randn(self.size) * self.sigma


class MemberA:
    r'''
        Initialize member with:
            initial_x: initial parameters
            rng_epsilons: random generator for future epsilons
            sigma: parameter to apply on mutations
    '''

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

        # if self._rank == 0:

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
        self.sigma = kwargs.get('sigma')

        a = True
        if False == True:
            # Init seedmatrix:
            self.rgn_init = random.Random(self.global_seed)

            self.seedmatrix_init_genes = np.zeros((self._size, self.workers_per_rank), dtype=np.int64)
            for i in range(self._size):
                row = self.rgn_init.sample(range(2**32 - 1), self.workers_per_rank)
                self.seedmatrix_init_genes[i] = row

            self.seedmatrix_init_epsilons = np.zeros((self._size, self.workers_per_rank), dtype=np.int64)
            for i in range(self._size):
                row = self.rgn_init.sample(range(2**32 - 1), self.workers_per_rank)
                self.seedmatrix_init_epsilons[i] = row

            # Init rng_epsilons_list
            self.rng_epsilons_list = [np.random.RandomState(self.seedmatrix_init_epsilons[self._rank, i]) for i in range(self.workers_per_rank)]

            # Init genes:
            self.init_genes = self.seedmatrix_init_genes[self._rank]

        elif True == True:

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
        # rng_epsilons=self.rng_epsilons_list[i]

        # self.members = [Member(init_parameters=self.initial_guess,
        #                       init_gene=self.init_genes[i],
        #                       rng_epsilons=self.rng_epsilons_list[i],
        #                       sigma=self.sigma)
        #                for i in range(self.workers_per_rank)]

    def apply_mutation(self, ranks_and_members_by_performance):
        """
        Preserve the top num_elite members, mutate the rest
        """
        no_elite_mask = ranks_and_members_by_performance > self.num_elite

        row, column = np.where(no_elite_mask)
        no_elite_tuples = tuple(zip(row, column))

        for rank, member_id in no_elite_tuples:
            if rank == self._rank:
                rng_genes = self.rng_genes_list[member_id]
                self.members[member_id].mutate(rng_genes)
