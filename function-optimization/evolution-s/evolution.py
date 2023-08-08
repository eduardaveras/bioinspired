import time
import numpy as np
import random as rd
from chromossome import Chromossome
from utils import calculate_time

# Evolution for solving any function for 30 dimensions
# tau = learning rate
# tau_prime = global learning rate

class Evolution:
    def __init__(self, n_iterations=200000, population_size=200,
                 dimensions=30, number_of_parents=200,
                 function="rosenbrock", crossover_rate=1,
                 learning_rate=2, global_learning_rate=2, epsilon=0.01,
                 crossover="intermediate", survivors_selection="plus", parents_selection="best"
                ):

        self.dimensions = dimensions
        self.function = function

        self.learning_rate = learning_rate
        self.global_learning_rate = global_learning_rate
        self.epsilon = epsilon

        self.best_fitness = None
        self.best_individual = None
        self.best_fitness_history = None
        self.best_individual_history = None

        self.population_size = population_size
        self.number_of_parents = number_of_parents
        self.n_iterations = n_iterations
        self.crossover_rate = crossover_rate
        self.crossover = crossover
        self.survivors_selection = survivors_selection
        self.parents_selection = parents_selection
        self.jump_chance = 0.0001

        self.population = []
        self.population_per_it = []
        self.iterations = 0

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

        return child

    def intermediate_crossover(self, parent1, parent2):
        child = Chromossome(self, mutation_step=np.zeros(self.dimensions), X=np.zeros(self.dimensions))
        for i in range(self.dimensions):
            child[i] = (parent1[i] + parent2[i]) / 2

        return child

    def select_parents(self):
        return rd.sample(self.population, k=self.number_of_parents)

    def select_best_parents(self):
        return sorted(self.population)[:self.number_of_parents]

    # We select the parents according to a uniform distribution of probabilities
    def select_parents_in_pair(self, parents):
        return rd.sample(parents, k=2)

    # We select the survivor wth (μ + λ) selection
    def filter_survivors_plus(self, parents, children, size):
        p = children + parents
        p.sort()

        return p[:size]

    # We select the survivor wth (μ , λ) selection
    def filter_survivors_comma(self, parents, children, size):
        p = children
        p.sort()

        for parent in parents:
            self.population.remove(parent)

        return p[:size]

    def end_condition(self):
        if sum(1 for x in self.population if x.isSolution) > 0:
            return False

        if self.iterations >= self.n_iterations or self.n_iterations == -1:
            return False

        return True

    def run(self):
        # Initialize
        self.population = self.init_population()
        self.population_per_it = []
        self.iterations = 0

        # criterio de parada
        while( self.end_condition() ):
            best_fitness = min([x.fitness() for x in self.population])
            best_individual = min(self.population , key=lambda x: x.fitness())
            print("Iteration: ", self.iterations)
            print("Best fitness: ", best_fitness)

            parents = []
            children = []

            self.population_per_it.append(self.population)
            self.iterations += 1
            # number_of_children / number_of_parents <= 7

            # Seleção de pais
            if self.parents_selection == "best":
                parents = self.select_best_parents()
            elif self.parents_selection == "random":
                parents = self.select_parents()
            else:
                raise Exception("Invalid parents selection type")

            while (len(children)/len(parents) < 7):
                # Seleciona os pais em pares para gerar os filhos
                parents_pair = self.select_parents_in_pair(parents)

                # Crossover
                if np.random.uniform(0, 1) <= self.crossover_rate:
                    if self.crossover == "intermediate":
                        child = self.intermediate_crossover(parents_pair[0], parents_pair[1])
                    elif self.crossover == "discrete":
                        child = self.discrete_crossover(parents_pair[0], parents_pair[1])
                    else:
                        raise Exception("Invalid crossover type")
                else:
                    child = parents_pair[0].copy()

                children.append(child)

            # Muta os filhos
            for child in children:
                child.mutate()

            # Seleção de sobreviventes
            if self.survivors_selection == "comma":
                self.population = self.filter_survivors_comma(parents, children, self.population_size)
            elif self.survivors_selection == "plus":
                self.population = self.filter_survivors_plus(parents, children, self.population_size)
            else:
                raise Exception("Invalid survivors selection type")


def main():
    e = Evolution()
    e.run()

if __name__ == "__main__":
    main()