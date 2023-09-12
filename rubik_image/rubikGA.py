from randomizer import Rand, Rand_cube, Rand_index, Rand_float
from cube import Cube
from utils import load_image, image_to_gray_pixels, gray_image_to_cubes, cubes_to_image, save_image

import numpy as np
import random
from datetime import datetime

class GA:
    def __init__(self, target_image_path="images/cabeca-gatinha-b.png", number_of_cubes_x=2, image_size=300, random_itens_size=10000):
        if image_size % number_of_cubes_x != 0:
            raise ValueError("Image size must be divisible by number_of_cubes")

        if (image_size / number_of_cubes_x) % 3 != 0:
            raise ValueError("Image size per Cube size must be divisible by 3")

        if image_size / number_of_cubes_x < 3:
            raise ValueError("Wrong values for number_of_cubes_x and image_size. Couldnt fit cubes in image!")

        self.number_of_cubes_x = number_of_cubes_x
        self.number_of_cubes = number_of_cubes_x * number_of_cubes_x
        self.cube_size = int(image_size / number_of_cubes_x)
        self.image_size = image_size
        self.target_image = load_image(target_image_path)
        self.target_gray = image_to_gray_pixels(self.target_image, self.image_size, self.cube_size//3)
        self.target_cubes = gray_image_to_cubes(self.target_gray)
        self.target_image_cubes = cubes_to_image(self.target_cubes, stroke_width=0, size_times=10, stroke_color=255)
        self.random_cubes = Rand_cube(random_itens_size)
        self.random_ints = Rand(0, self.number_of_cubes - 1, random_itens_size)
        self.random_floats = Rand_float(random_itens_size)
        self.random_indexes = Rand_index(0, number_of_cubes_x - 1, random_itens_size)

        self.population = []
        self.best_cube = None
        self.parents = None
        self.children = None
        self.iteration = 0
        self.solution_was_found = False


    def get_best_individual(self, list_=None):
        if list_ == None:
            list_ = self.population

        return min(self.population)

    def get_random_individuals(self, n, list_=None):
        if list_ == None:
            list_ = self.population

        return random.sample(list_, n)

    def get_best_individuals(self, n, list_=None):
        if list_ == None:
            list_ = self.population

        return sorted(list_)[:n]

    def get_worst_individuals(self, n, list_=None):
        if list_ == None:
            list_ = self.population

        return sorted(list_, reverse=True)[:n]

    def set_population(self, population):
        self.population = population

    def init_population(self):
        for _ in range(self.number_of_cubes):
            new_individual = Individual(self)
            self.population.append(new_individual)

    # parent selection
    def parent_tournament(self, tournament_size, parents_size):
        list_parents = self.get_random_individuals(tournament_size)
        #select the best {parents_size} parents
        best_parents = self.get_best_individuals(parents_size, list_=list_parents)
        #print(f"selected parent fitness: {[i.fitness for i in list_parents]}", end='\n', sep=', ')
        print(f"best parent fitness: {[i.fitness for i in best_parents]}",  end='\n', sep=', ')

        return best_parents

    # survivor selection
    def choose_survivors(self, size):
        bests = self.get_best_individuals(size, list_=self.population+self.children)

        self.population = bests

    def single_crossover(self, parent1, parent2):
        #new individual
        child = Individual(self)

        for i in range (parent1.cubes.shape[0]):
            for j in range(parent1.cubes.shape[1]):
                if self.random_floats.pop() < 0.5:
                    child.cubes[i][j] = parent1.cubes[i][j]
                else:
                    child.cubes[i][j] = parent2.cubes[i][j]

        return child

    def combine_crossover(self, parents):
        # take the mean of the parents (cubes)
        child = Individual(self)

        for parent in parents:
            child.cubes += parent.cubes

        #child.cubes /= len(parents)

        return child

    def finish_condition(self, max_iterations):
        solutions = [ indiv.isSolution for indiv in self.population]
        #print("We found " + str(solutions.count(True)) + " solutions")

        if True in solutions and max_iterations != -1:
            print(f"Found the solution, stopping the algorithm...")
            self.solution_was_found = True
            return True

        if self.iteration == max_iterations:
            print(f"Reached the max iterations, stopping the algorithm...")
            return True

        return False

    def switch_indiv(self, indiv1, indiv2):
        self.population.remove(indiv1)
        self.population.append(indiv2)

    def save_iteration(self):
        new_iteration_dict = {}
        new_iteration["children"] = self.children
        new_iteration["parents"] = self.parents

        self.iterations[self.iteration] = new_iteration_dict
        pass

    def run(self, population_size=500, tournament_size=10, number_of_children=2, number_of_parents=4, recombination_rate=0.5, recombination_method="combine", mutation_rate=0.5, max_iterations=10000):
        # Init population

        self.init_population()

        while not self.finish_condition(max_iterations):
            self.iteration += 1
            print(f"{self.iteration}º iteration -----------\n")
            #print(f"Population fitness: {[i.fitness for i in self.population]}", end='\n', sep=', ')

            # Parent selection
            self.parents = self.parent_tournament(tournament_size, number_of_parents)

            # Recombination
            self.children = []
            for _ in range(number_of_children):
                if self.random_floats.pop() < recombination_rate:
                    #print(f"Recombining parents {self.parents[0].fitness} and {self.parents[1].fitness}"
                    child = self.single_crossover(self.parents[0], self.parents[1])
                    self.children.append(child)
                    print(f"single crossover!")
                else:
                    child = self.combine_crossover(self.parents)
                    self.children.append(child)
                    print(f"combined crossover!")

            print(f"Children fitness: {[i.fitness for i in self.children]}", end='\n', sep=', ')

            # Mutation
            for child in self.children:
                if self.random_floats.pop() < mutation_rate:
                    print(f"Mutating child {child.fitness}")
                    child.mutate()
                    print(f"Mutated child fitness: {child.fitness}")

            # Survivor selection
            self.choose_survivors(population_size)


            print(f"Best fitness: {self.get_best_individual().fitness}\n")
        return self



class Individual:
    def __init__(self, ga_instance, cubes=np.array([[]]), empty_init=False):
        self.fitness = 0
        self.number_of_cubes_x = ga_instance.number_of_cubes_x
        self.number_of_cubes = self.number_of_cubes_x * self.number_of_cubes_x
        self.random_cubes = ga_instance.random_cubes
        self.random_ints = ga_instance.random_ints
        self.random_indexes = ga_instance.random_indexes
        self.target = ga_instance.target_cubes
        self.isSolution = False
        self.empty_init = empty_init

        if type(cubes) is not np.ndarray:
            raise TypeError("cubes initializer must be a numpy array")
        elif cubes.shape == (1, 0):
            self.random_init()
        elif cubes.shape != (self.number_of_cubes_x, self.number_of_cubes_x):
            raise ValueError("cubes initializer must be a numpy array with shape (number_of_cubes_x, number_of_cubes_x)")
        else:
            self.cubes = cubes

        self.evaluate_fitness()

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def __gt__(self, other):
        return self.fitness > other.fitness

    def __le__(self, other):
        return self.fitness <= other.fitness

    def __ge__(self, other):
        return self.fitness >= other.fitness

    def set_cubes(self, cubes):
        self.cubes = cubes
        self.evaluate_fitness()

    def evaluate_fitness(self):
        max_fitness = self.number_of_cubes_x * self.number_of_cubes_x * 255 * 9

        for i in range(0, self.number_of_cubes_x):
            for j in range(0, self.number_of_cubes_x):
                self.fitness += np.sum(np.abs(self.cubes[i][j].gray_map - self.target[i][j]))

        self.fitness /= max_fitness

        if (self.fitness <= 0.1):
            self.isSolution = True

    def random_init(self):
        # Create an np.array of random cubes with the type Cube
        # and then create a list of cubes
        # from that array
        if not self.state:
            cubes = np.zeros((self.number_of_cubes_x, self.number_of_cubes_x), dtype=Cube)

            for i in range(self.number_of_cubes_x):
                for j in range(self.number_of_cubes_x):
                    cubes[i][j] = self.random_cubes.pop()

            self.set_cubes(cubes)
        else:
            # create an np.array of empty cubes
            cubes = np.zeros((self.number_of_cubes_x, self.number_of_cubes_x), dtype=Cube)
            self.set_cubes(cubes)

    # mutate n cubes of one individual
    def mutate(self):
        indexes = self.random_indexes.pop()
        print(f"Mutating indexes: {indexes}")
        for index in indexes:
            i, j = index
            self.cubes[i][j] = self.random_cubes.pop()

        self.evaluate_fitness()

    def to_image(self, size_times=1, stroke_width=0, stroke_color=255):
        cubes = self.cubes

        # Assume the first cube's image shape is (rows, columns, channels)
        cube_height, cube_width, _ = cubes[0, 0].image().shape

        image = np.zeros((cubes.shape[0]*cube_height*size_times, cubes.shape[1]*cube_width*size_times, 3), dtype=np.uint8)

        for i in range(0, image.shape[0], cube_height*size_times):
            for j in range(0, image.shape[1], cube_width*size_times):
                cube = cubes[i//(cube_height*size_times), j//(cube_width*size_times)]
                cube_times = np.kron(cube.image(), np.ones((size_times, size_times, 1), dtype=np.uint8))

                # Apply stroke to cube_times
                if stroke_width > 0:
                    cube_times[0:stroke_width, :, :] = stroke_color
                    cube_times[-stroke_width:, :, :] = stroke_color
                    cube_times[:, 0:stroke_width, :] = stroke_color
                    cube_times[:, -stroke_width:, :] = stroke_color

                start_row = i
                start_col = j
                end_row = start_row + cube_times.shape[0]
                end_col = start_col + cube_times.shape[1]
                image[start_row:end_row, start_col:end_col, :] = cube_times

        return image

if __name__ == "__main__":
    ga = GA()
    ga.init_population()
    ga.population[0].cubes[0][0] + ga.population[0].cubes[1][0]