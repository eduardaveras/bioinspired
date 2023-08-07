import random
from datetime import datetime
from chromossome import Chromossome
import numpy as np

class Genetic:
    def __init__(self, population_size=100, max_iterations=10000,
                 pair_children_size=3, dimensions=30,
                 recombination_method="", recombination_probability=0.9,
                 mutation_method="", mutation_probability=0.7,
                 parent_method="tournament", survivor_method="best",
                 ):


        self.solution_was_found = False
        self.population_size = population_size
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.population = []
        self.max_iterations = max_iterations
        self.iterations = 0
        self.dimensions = dimensions
        self.iteration_info = []
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.pair_children_size = pair_children_size
        # Methods
        self.parent_method = parent_method
        self.survivor_method = survivor_method
        self.recombination_method = recombination_method
        self.mutation_method = mutation_method

    def run(self):
        random.seed(datetime.now().timestamp())
        """
        begin
            initialise population with random candidate solution;
            evaluate each candidate;
            repeat until (termination condition is satisfied) do
                1 select parents;
                2 recombine pairs of parents
                3 mutate the resulting offspring;
                4 evaluate new candidates;
                5 select individuals for the next generation;
            od
        end
        """
        # initilize population
        self.population = self.init_population()

        while not self.finish_condition(self.population):
            self.iterations += 1
            children = []
            parents = []
            print(f"\n{self.iterations} iteration ---------\n")
            #print(f"Population fitnesses: {[i.fitness() for i in self.population]}", end='\n', sep=',')
            print(f"best fitness: {min([i.fitness() for i in self.population])}")

            # Selection of parents
            for _ in range(self.pair_children_size):
                if self.parent_method == "tournament":
                    parents = self.parent_tournament(5, 2)
                elif self.parent_method == "spinwheel":
                    parents = self.parent_spinwheel(2)
                else:
                    raise Exception("Invalid parent method")

                # Recombination
                if random.choices([True, False], weights=[self.recombination_probability, 1-self.recombination_probability], k=1)[0]:
                    print(f"Recombining parents => ", end=' ')

                    # Recombination occurs
                    child1, child2 = self.recombine_line(parents[0], parents[1])
                    children.append(child1)
                    children.append(child2)

                    print(f"Children from crossover: {[(i.fitness()) for i in children]}", end='\n', sep=',')
                else:
                    children.append(parents[0])
                    children.append(parents[1])
                    print(f"Same as the parents: {[i.fitness() for i in children]}" ,end='\n', sep=' ')

                for indiv in children:
                    self.population.append(indiv)
                    if random.choices([True, False], weights=[self.mutation_probability, 1-self.mutation_probability], k=1)[0]:
                        # Mutating
                        indiv_mutated = self.mutate_non_uniform(indiv, 5000)
                        # if(indiv_mutated.fitness() < indiv.fitness()):
                        self.switch_indiv(indiv, indiv_mutated)

            # Survivor selection
            self.choose_survivor(self.population_size)

            # # self.population = sorted(self.population, key=lambda indiv: indiv.fitness(), reverse=True)
            self.iteration_info.append([i.fitness() for i in self.population])

            print("---------------------------")

    def switch_indiv(self, indiv1, indiv2):
        self.population.remove(indiv1)
        self.population.append(indiv2)

    def finish_condition(self, population):
        solutions = [indiv.isSolution for indiv in population]
        print("We found " + str(solutions.count(True)) + " solutions")

        if True in solutions and self.max_iterations != -1:
            print(f"Found the solution, ending...")
            self.solution_was_found = True
            return True

        if False not in solutions:
            print(f"All population is solution, ending...")
            self.solution_was_found = True
            return True

        if self.iterations == self.max_iterations:
            print("Max iteration number was reached")
            return True

        return False

    def recombine_line(self, parent1, parent2, alpha=0.5):
        child1 = parent1.copy()
        child2 = parent2.copy()

        for i in range(len(parent1.X)):
            I = alpha * (max(parent1.X[i], parent2.X[i]) - min(parent1.X[i], parent2.X[i]))
            child1.X[i] = random.uniform(min(parent1.X[i], parent2.X[i]) - I, max(parent1.X[i], parent2.X[i]) + I)
            child2.X[i] = random.uniform(min(parent1.X[i], parent2.X[i]) - I, max(parent1.X[i], parent2.X[i]) + I)

        return child1, child2

    def mutate_non_uniform(self, individual, max_gen, b=5):
        t = self.iterations / max_gen
        index = random.randint(0, len(individual.X) - 1)
        delta = (individual.function.bounds[1] - individual.function.bounds[0]) * (1 - np.random.rand()**(1 - t)**b)
        if random.random() < 0.5:
            individual.X[index] -= delta
        else:
            individual.X[index] += delta
        return individual

    def init_population(self):
        population = []
        for i in range(self.population_size):
            population.append(Chromossome(dimensions=self.dimensions))

        return population

    def parent_tournament(self, choices_size, return_size):
        population = self.population
        list_parents = random.sample(population, choices_size)
        best_parents = sorted(list_parents, key=lambda indiv: indiv.fitness(), reverse=False)
        print(f"Selecting parents: {[i.fitness() for i in best_parents]}", end='\n', sep=',')
        # Best parents:
        print(f"Chosen parents: {[i.fitness() for i in best_parents][:return_size]}", end='\n', sep=',')

        return best_parents[:return_size]

    def parent_spinwheel(self, return_size):
        population = self.population
        parents = []

        for _ in range(return_size):
            fitness_sum = sum([indiv.fitness() for indiv in population])
            fitnesses = [indiv.fitness()/fitness_sum for indiv in population]

            choice = random.choices(population, weights=fitnesses, k=1)[0]

            parents.append(choice)
            population.remove(choice)

        return parents

    def choose_survivor(self, return_size):
        bests = sorted(self.population, key=lambda indiv: indiv.fitness(), reverse=False)

        for _ in range(len(self.population) - return_size):
            print("Removing worst: " + str(bests[-1].fitness()))
            self.population.remove(bests[-1])
            bests.remove(bests[-1])

g = Genetic()
# g.run()
g.population = g.init_population()
g.run()

