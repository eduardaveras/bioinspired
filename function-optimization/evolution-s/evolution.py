import time
import numpy as np
import random as rd
from chromossome import Chromossome

# Evolution for solving any function for 30 dimensions
# tau = learning rate
# tau_prime = global learning rate

class Evolution:
    def __init__(self, n_iterations=200000, population_size=50,
                 dimensions=30, number_of_parents=50,
                 function="rosenbrock", crossover_rate=1,
                 learning_rate=2.1, global_learning_rate=2.1, epsilon=0.01,
                 selection_pressure=7,
                 crossover="intermediate", survivors_selection="plus", parents_selection="random"
                ):

        self.dimensions = dimensions
        self.function = function

        self.learning_rate = learning_rate
        self.global_learning_rate = global_learning_rate
        self.epsilon = epsilon

        self.best_fitness = None
        self.best_individual = None
        self.best_fitness_history = []
        self.solution_was_found = False

        self.population_size = population_size
        self.number_of_parents = number_of_parents
        self.n_iterations = n_iterations
        self.crossover_rate = crossover_rate
        self.crossover = crossover
        self.survivors_selection = survivors_selection
        self.parents_selection = parents_selection
        self.selection_pressure = selection_pressure
        self.jump_chance = 0.01

        self.population = []
        self.iteration_info = []
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
            self.solution_was_found = True
            return False

        if self.iterations >= self.n_iterations or self.n_iterations == -1:
            return False

        return True

    def run(self):
        # Initialize
        self.population = self.init_population()
        self.iterations = 0

        # criterio de parada
        while( self.end_condition() ):
            best_fitness = min([x.fitness() for x in self.population])
            self.best_fitness_history.append(best_fitness)
            best_individual = min(self.population , key=lambda x: x.fitness())
            print("Iteration: ", self.iterations)
            print("Best fitness: ", best_fitness)

            parents = []
            children = []

            self.iterations += 1
            # number_of_children / number_of_parents <= 7

            # before_parents_selection = self.parents_selection
            # if self.best_fitness_history.count(self.best_fitness_history[-1]) == 50:
            #         print("Changing!")
            #         self.survivors_selection = "comma"
            #         self.parents_selection = "random"

            # Seleção de pais
            if self.parents_selection == "best":
                parents = self.select_best_parents()
            elif self.parents_selection == "random":
                parents = self.select_parents()
            else:
                raise Exception("Invalid parents selection type")

            while (len(children)/len(parents) < self.selection_pressure):
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
            elif self.survivors_selection == "plus_comma":
                if best_individual.fitness() < 2.0:
                    self.population = self.filter_survivors_plus(parents, children, self.population_size) + [best_individual]
                else:
                    self.population = self.filter_survivors_comma(parents, children, self.population_size) + [best_individual]
            else:
                raise Exception("Invalid survivors selection type")

            self.iteration_info.append([i.fitness() for i in self.population])


def main():
    e = Evolution()
    e.run()

if __name__ == "__main__":
    main()