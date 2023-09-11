from randomizer import Rand, Rand_cube
from cube import Cube
from utils import load_image, image_to_gray_pixels, gray_image_to_cubes, cubes_to_image, save_image

import numpy as np
import random
from datetime import datetime

class GA:
    def __init__(self, target_image_path="images/monalisa.jpg", n_cubes_x=4, image_size=300, random_itens_size=10000, max_iterations=1000):
        if image_size % n_cubes_x != 0:
            raise ValueError("Image size must be divisible by n_cubes")

        if (image_size / n_cubes_x) % 3 != 0:
            raise ValueError("Image size per Cube size must be divisible by 3")

        if image_size / n_cubes_x < 3:
            raise ValueError("Wrong values for n_cubes_x and image_size. Couldnt fit cubes in image!")

        self.n_cubes_x = n_cubes_x
        self.n_cubes = n_cubes_x * n_cubes_x
        self.cube_size = int(image_size / n_cubes_x)
        self.image_size = image_size
        self.population = []
        self.target_image = load_image(target_image_path)
        self.target_gray = image_to_gray_pixels(self.target_image, self.image_size, self.cube_size//3)
        self.target_cubes = gray_image_to_cubes(self.target_gray)
        self.target_image_cubes = cubes_to_image(self.target_cubes, stroke_width=1, size_times=5, stroke_color=255)
        self.random_cubes = Rand_cube(random_itens_size)
        self.random_ints = Rand(0, 5, random_itens_size)
        self.random_ints_cubes = Rand(0, n_cubes_x * n_cubes_x - 1, random_itens_size)

        self.best_cube = None
        self.iteration = 0
        self.max_iterations = max_iterations
        self.tournament_size = 5
        self.num_children = 2
        self.recombination_rate = 0.7
        self.mutation_rate = 0.3
        self.solution_was_found = False

    def gest_best_individual(self):
        return max(self.population)

    def get_best_individuals(self, n):
        return sorted(self.population, reverse=True)[:n]

    def get_worst_individuals(self, n):
        return sorted(self.population, reverse=False)[:n]

    def set_population(self, population):
        self.population = population

    def init_population(self):
        for _ in range(self.n_cubes):
            new_individual = Individual(self)
            self.population.append(new_individual)

    # parent selection (ok)
    def parent_tournament(self, parents_number):
        list_parents = random.sample(self.population, self.tournament_size)
        #select the best two parents
        best_parents = sorted(list_parents, reverse=True)[:parents_number]
        print(f"selected parent fitness: {[i.fitness for i in list_parents]}", end='\n', sep=', ')
        print(f"best parent fitness: {[i.fitness for i in best_parents]}",  end='\n', sep=', ')

        return best_parents

    # survivor selection (ok)
    def choose_survivors(self, size):
        bests = sorted(self.population, reverse=True)[:size]
        
        for _ in range(len(self.population) - size):
            print("Removing worst: " + str(self.population[-1].fitness))
            self.population.remove(bests[-1])
            bests.remove(bests[-1])


    def matrix_crossover(self, parent1, parent2):
        children = []

        for _ in range(self.num_children):
            #new individual
            child = Individual(self)
            
            for i in range (parent1.cubes.shape[0]):
                for j in range(parent1.cubes.shape[1]):
                    if random.random() < 0.5:
                        child.cubes[i][j] = parent1.cubes[i][j]
                    else:
                        child.cubes[i][j] = parent2.cubes[i][j]

            children.append(child)

        return children


    # mutate n cubes of the population
    def mutation(self, individual):
        n_cubes = self.random_ints_cubes.pop()
        #print(n_cubes)
        cubes_index = random.sample(range(self.n_cubes), n_cubes)
        #print(cubes_index)
        #mutate the cubes
        for cube in cubes_index:
            #print(cube//self.n_cubes_x, cube%self.n_cubes_x)
            individual.cubes[cube//self.n_cubes_x][cube%self.n_cubes_x] = self.random_cubes.pop()

        individual.evaluate_fitness(self.target_cubes)

        return individual
    
    def finish_condition(self):
        solutions = [ indiv.isSolution for indiv in self.population]
        print("We found " + str(solutions.count(True)) + " solutions")
        
        if True in solutions and self.max_iterations != -1:
            print(f"Found the solution, stopping the algorithm...")
            self.solution_was_found = True
            return True
        
        if self.iteration == self.max_iterations:
            print(f"Reached the max iterations, stopping the algorithm...")
            return True
        
        return False
    
    def switch_indiv(self, indiv1, indiv2):
        self.population.remove(indiv1)
        self.population.append(indiv2)

    def run(self):
        # Init population
        self.init_population()
        
        while not self.finish_condition():
            self.iteration += 1
            children = []
            parents = []
            print(f"\n{self.iteration}º iteration -----------\n")
            print(f"Population fitness: {[i.fitness for i in self.population]}", end='\n', sep=', ')
            
            # Parent selection
            parents = self.parent_tournament(2)
            
            # Recombination
            if random.choices([True, False], weights=[self.recombination_rate, 1-self.recombination_rate], k=1)[0]:
                print(f"Recombining parents {parents[0].fitness} and {parents[1].fitness}")
                children = self.matrix_crossover(parents[0], parents[1])
                
                print(f"Children from crossover: {[i.fitness for i in children]}", end='\n', sep=', ')
                
            else:
                children = parents[:self.num_children]
                print(f"Same as the parents: {[i.fitness for i in children]}", end='\n', sep=', ')
                
            # Mutation
            for ind in children:
                self.population.append(ind)
                
                if random.choices([True, False], weights=[self.mutation_rate, 1-self.mutation_rate], k=1)[0]:
                    print(f"Mutating child {ind.fitness}")
                    
                    ind_mutated = self.mutation(ind)
                    print(f"Mutated child fitness: {ind_mutated.fitness}")
                    self.switch_indiv(ind, ind_mutated)
                    
            # Survivor selection
            self.choose_survivors(len(self.population))
            
            print("-----------------------")
                    
        return self



class Individual:
    def __init__(self, ga_instance, cubes=np.array([[]])):
        self.fitness = 0
        self.n_cubes_x = ga_instance.n_cubes_x
        self.random_cubes = ga_instance.random_cubes
        self.isSolution = False

        if type(cubes) is not np.ndarray:
            raise TypeError("cubes initializer must be a numpy array")
        elif cubes.shape == (1, 0):
            self.random_init()
        elif cubes.shape != (self.n_cubes_x, self.n_cubes_x):
            raise ValueError("cubes initializer must be a numpy array with shape (n_cubes_x, n_cubes_x)")
        else:
            self.cubes = cubes


        self.evaluate_fitness(ga_instance.target_cubes)

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __lt__(self, other):
        return self.fitness > other.fitness

    def __gt__(self, other):
        return self.fitness < other.fitness

    def __le__(self, other):
        return self.fitness >= other.fitness

    def __ge__(self, other):
        return self.fitness <= other.fitness

    def evaluate_fitness(self, target_cubes):
        max_fitness = self.n_cubes_x * self.n_cubes_x * 255 * 9

        for i in range(0, self.n_cubes_x):
            for j in range(0, self.n_cubes_x):
                self.fitness += np.sum(np.abs(self.cubes[i][j].gray_map - target_cubes[i][j]))

        self.fitness /= max_fitness

        if (self.fitness <= 0.1):
            self.isSolution = True

    def random_init(self):
        # Create an np.array of random cubes with the type Cube
        # and then create a list of cubes
        # from that array
        cubes = np.zeros((self.n_cubes_x, self.n_cubes_x), dtype=Cube)

        for i in range(self.n_cubes_x):
            for j in range(self.n_cubes_x):
                cubes[i][j] = self.random_cubes.pop()

        self.cubes = cubes


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

    ga.run()
    