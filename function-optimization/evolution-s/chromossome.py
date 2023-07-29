import numpy as np
from functions import Functions

class Chromossome:
    def __init__(self, dimensions=30, mutation_step=None, X=None, global_learning_rate=1, learning_rate=1, epsilon=0.01, function_name="ackley"):
        self.dimensions = dimensions
        self.learning_rate = (1 / np.sqrt(2 * np.sqrt(dimensions))) * learning_rate
        self.global_learning_rate = (1 / np.sqrt(2 * dimensions)) * global_learning_rate
        self.epsilon = epsilon
        self.function = Functions(function_name, dimensions)

        self.X = X
        self.mutation_step = mutation_step

        if X is None:
            self.X, self.mutation_step = self.random_chromossome()

    def __len__(self):
        return self.dimensions

    def __getitem__(self, index):
        if isinstance(index, int) and index < self.dimensions and index >= 0:
            return self.X[index], self.mutation_step[index]
        else:
            raise Exception("Index out of bounds")

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
        return self.function(self.X)
