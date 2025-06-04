from abc import ABC, abstractmethod

class World:

    @abstractmethod
    def evaluate_individual(self, genotype) -> float:
        pass

    @abstractmethod
    def geno2pheno(self, genotype) -> object:
        pass
