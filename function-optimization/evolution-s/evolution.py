import numpy as np
import random as rd
from chromossome import Chromossome

# Evolution for solving any function for 30 dimensions
# tau = learning rate
# tau_prime = global learning rate

class Evolution:
    def __init__(self, n_iterations=100, population_size=10, dimensions=30,
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

        self.n_iterations = 100

        self.population = self.init_population()

    def init_population(self):
        population = []

        for _ in range(self.population_size):
            new_chromossome = Chromossome(self)
            population.append(new_chromossome)

        return population

    def intermediate_recombination(self, parent1, parent2):
        return (parent1 + parent2) / 2

    def intermediate_index_recombination(self, index, parent1, parent2):
        return (parent1[index] + parent2[index]) / 2

    # We recombine the population with a discrete crossover or intermediate crossover
    def discrete_crossover(self, parent1, parent2):
        child = Chromossome(self, np.zeros(self.dimensions), np.zeros(self.dimensions))
        for i in range(self.dimensions):
            if np.random.uniform(0, 1) < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]

        child.set_parents([parent1, parent2])
        return child

    def intermediate_crossover(self, parent1, parent2):
        child = np.zeros(self.dimensions)
        for i in range(self.dimensions):
            child[i] = (parent1[i] + parent2[i]) / 2

        child.set_parents([parent1, parent2])
        return child

    # We select the parents according to a uniform distribution of probabilities
    def select_parents(self):
        return rd.choices(self.population, k=2)

    # We select the survivor wth (μ + λ) selection
    def filter_survivors_plus(self, parents, children, size):
        p = parents + children
        p_sorted = p.sort(reverse=True)
        remove_size = len(p_sorted) - size

        for _ in range(remove_size):
            p.remove(p_sorted.pop())

        return p

    # We select the survivor wth (μ , λ) selection
    def filter_survivors_comma(self, parents, children, size):
       pass 
   # Duvida aqui???

