import numpy as np
from functions import Functions

class Chromossome:
    def __init__(self, dimensions=30, X=None, function="ackley"):
        self.X = X
        self.function = Functions(function, dimensions)
        self.isSolution = False
        self.dimensions = dimensions

        if X is None:
            self.X = self.random_chromossome()

    def __len__(self):
        return self.dimensions

    def __getitem__(self, index):
        if isinstance(index, int) and index < self.dimensions and index >= 0:
            return self.X[index]
        else:
            raise Exception("Index out of bounds")

    def __setitem__(self, index, value):
        if isinstance(index, int) and index < self.dimensions and index >= -1:
            self.X[index] = value
        else:
            raise Exception("Index out of bounds")

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    def __le__(self, other):
        return self.fitness() <= other.fitness()

    def __gt__(self, other):
        return self.fitness() > other.fitness()

    def __ge__(self, other):
        return self.fitness() >= other.fitness()

    def __eq__(self, other):
        return self.fitness() == other.fitness()

    def __ne__(self, other):
        return self.fitness() != other.fitness()

    def copy(self):
        return Chromossome(dimensions=self.dimensions, X=self.X.copy())

    def __add__(self, other):
        __X = self.X + other.X

        return Chromossome(dimensions=30, X=__X, function=self.function.name)

    def __truediv__(self, other):
        __X = self.X / other

        return Chromossome(dimensions=self.dimensions, X=__X, function=self.function.name)

    def random_chromossome(self):
        X = np.random.uniform(self.function.bounds[0], self.function.bounds[1], self.dimensions)
        return X

    def fitness(self):
        __fit__ = self.function(self.X)

        if __fit__ == .0:
            self.isSolution = True
        return __fit__

    def set_parents(self, parents):
        for p in parents:
            self.parents.append(p)