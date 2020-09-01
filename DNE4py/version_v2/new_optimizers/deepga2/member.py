import numpy as np


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
