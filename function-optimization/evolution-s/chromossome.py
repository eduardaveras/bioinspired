import numpy as np
from functions import Functions

class Chromossome:
    def __init__(self, Evolution, mutation_step=None, X=None):
        self.Evolution = Evolution
        self.dimensions = Evolution.dimensions
        self.epsilon = Evolution.epsilon

        self.learning_rate = (1 / np.sqrt(2 * np.sqrt(self.dimensions))) * Evolution.learning_rate
        self.global_learning_rate = (1 / np.sqrt(2 * self.dimensions)) * Evolution.global_learning_rate
        self.function = Functions(Evolution.function, self.dimensions)

        self.X = X
        self.mutation_step = mutation_step
        self.parents = []
        self.isSolution = False

        if X is None:
            self.X, self.mutation_step = self.random_chromossome()

    def __len__(self):
        return self.dimensions

    def __getitem__(self, index):
        if isinstance(index, int) and index < self.dimensions and index >= 0:
            return np.array([self.X[index], self.mutation_step[index]])
        else:
            raise Exception("Index out of bounds")

    def __setitem__(self, index, value):
        if isinstance(index, int) and index < self.dimensions and index >= 0:
            self.X[index] = value[0]
            self.mutation_step[index] = value[1]
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
        return Chromossome(self.Evolution, self.mutation_step.copy(), self.X.copy())

    def __add__(self, other):
        X = self.X + other.X
        mutation_step = self.mutation_step + other.mutation_step

        return Chromossome(self.Evolution, mutation_step, X)

    def __truediv__(self, other):
        X = self.X / other
        mutation_step = self.mutation_step / other

        return Chromossome(self.Evolution, mutation_step, X)

    def random_chromossome(self):
        X = np.random.uniform(self.function.bounds[0], self.function.bounds[1], self.dimensions)
        mutation_step = np.random.uniform(0, 1, self.dimensions)
        return X, mutation_step

    def mutate(self):
        for i in range(self.dimensions):
            self.mutation_step[i] = self.mutation_step[i] * np.exp(self.global_learning_rate * np.random.normal(0, 1) + self.learning_rate * np.random.normal(0, 1))
            self.X[i] = self.X[i] + self.mutation_step[i] * np.random.normal(0, 1)

            # Check bounds
            if self.mutation_step[i] < self.epsilon:
                self.mutation_step[i] = self.epsilon

            if self.X[i] < self.function.bounds[0]:
                self.X[i] = self.function.bounds[0]
            elif self.X[i] > self.function.bounds[1]:
                self.X[i] = self.function.bounds[1]

    def fitness(self):
        __fit__ = self.function(self.X)

        if __fit__ == .0:
            self.isSolution = True
        return __fit__

    def set_parents(self, parents):
        for p in parents:
            self.parents.append(p)
