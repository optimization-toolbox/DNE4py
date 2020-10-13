import numpy as np


class Member:
    r'''
        Initialization:
            initial_phenotype: initial list of parameters
            initial_genotype: initial seed and initial sigma

        Internal attributes:
            genotype: list of seeds
            phenotype: list of parameters

        Methods:
            recreate(new_genotype):
                update genotype and phenotype from new_genotype
            mutate(rng_genes):
                create a new gene and update genotype and phenotype
    '''

    def __init__(self, initial_phenotype, genotype):

        # Define initial phenotype:
        self.initial_phenotype = initial_phenotype
        self.phenotype = self.initial_phenotype.copy()

        # Internal attributes:
        self.rng = np.random.RandomState()
        self.size = len(self.initial_phenotype)

        # Define genotype and phenotype:
        self.recreate(genotype)

    def mutate(self, rng_genes, sigma):
        r'''create a new gene and update genotype and phenotype'''

        # Increase genotype:
        seed = rng_genes.randint(0, 2 ** 32 - 1)
        self.genotype.append([seed, sigma])

        # Mutate phenotype:
        self.rng.seed(seed)
        self.phenotype += self.rng.randn(self.size) * sigma


    def recreate(self, new_genotype):
        r'''update genotype and phenotype from new_genotype'''

        # Set genotype:
        self.genotype = new_genotype[:]

        # Set phenotype:
        self.phenotype[:] = self.initial_phenotype[:]
        for seed, sigma in self.genotype:
            self.rng.seed(seed)
            self.phenotype += self.rng.randn(self.size) * sigma