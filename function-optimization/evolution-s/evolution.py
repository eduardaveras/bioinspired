import numpy as np
from chromossome import Chromossome

# Evolution for solving any function for 30 dimensions
# tau = learning rate
# tau_prime = global learning rate

class Evolution:
    def __init__(self, population_size=1000, dimensions=30,
                 function="ackley", crossover_rate=0.2, mutation_rate=0.2,
                 learning_rate=1, global_learning_rate=1, epsilon=0.01):

        self.population_size = population_size
        self.dimensions = dimensions
        self.function = function

        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

        self.learning_rate = learning_rate
        self.global_learning_rate = global_learning_rate
        self.epsilon = epsilon

        self.best_fitness = None
        self.best_individual = None
        self.best_fitness_history = None
        self.best_individual_history = None

        self.population = self.init_population()

    def init_population(self):
        population = []

        for _ in range(self.population_size):
            new_chromossome = Chromossome(dimensions=self.dimensions, global_learning_rate=self.global_learning_rate,
                                            learning_rate=self.learning_rate, epsilon=self.epsilon, function_name=self.function)
            population.append(new_chromossome)

        return population


    # We evaluate the fitness of each individual
    def evaluate_fitness(self):
        for i in range(self.population_size):
            self.fitness[i] = self.function(self.population[i])


    # We recombine the population with a discrete crossover or intermediate crossover
    def discrete_crossover(self, parent1, parent2):
        child = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            if np.random.uniform(0, 1) < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        return child

    def intermediate_crossover(self, parent1, parent2):
        child = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            child[i] = (parent1[i] + parent2[i]) / 2

        return child

    # We select the parents with a uniform distribution
    def select_parents(self):
        parent1 = np.random.randint(0, self.population_size)
        parent2 = np.random.randint(0, self.population_size)

        return parent1, parent2

    # We select the survivor wth (μ + λ) selection
    def select_survivor_plus(self, parent, child, size):
        total_population = np.concatenate((parent, child))
        total_population.sort(key=lambda ind: self.fitness[ind])

        return total_population[:size]

    # We select the survivor with (μ, λ) selection
    def select_survivor_comma(self, child, size):
        child.sort(key=lambda ind: self.fitness[ind])

        return child[:size]




