from cube import Cube
from image_utils import *
import random

class cubesToImage:
    def __init__(self, target_image_path="images/cabeca-gatinha-b.png", n_cubes_x=5, image_size=450, random_itens_size=10000):
        if image_size % n_cubes_x != 0:
            raise ValueError("Image size must be divisible by number_of_cubes")

        if (image_size / n_cubes_x) % 3 != 0:
            raise ValueError("Image size per Cube size must be divisible by 3")

        if image_size / n_cubes_x < 3:
            raise ValueError("Wrong values for n_cubes_x and image_size. Couldnt fit cubes in image!")

        self.image_size = image_size
        self.n_cubes_x = n_cubes_x
        self.cube_size = image_size // n_cubes_x
        self.target_image = load_image(target_image_path)
        self.target_gray_nearest = gray_nearest(image_to_gray_pixels(self.target_image, self.image_size, self.cube_size//3))
        self.target_color_cubes = image_gray_to_image_cubes_color(self.target_gray_nearest)
        self.target_cubes = gray_image_to_cubes(self.target_gray_nearest)
        self.target_image_gray_cubes = cubes_to_image(self.target_cubes, stroke_width=0, size_times=10, stroke_color=255)

class GA:
    def __init__(self, target_cube, population_size=10):
        self.target_cube = target_cube

        self.population = []
        self.population_size = population_size

        self.crossover_rate = 0.7
        self.mutation_rate = 0.01
        self.elitism_rate = 0.1
        self.max_generations = 1000
        self.max_fitness = 1
        self.min_fitness = 0
        self.fitness = []
        self.best_individual = None
        self.best_fitness = 0

    def init_population(self):
        for _ in range(0, self.population_size):
            cube = Cube(self.target_cube)
            self.population.append(cube)
            
    def parent_selection(self, tournament_size, n_parents):
        list_parents = random.sample(self.population, tournament_size)
        best_parents = sorted(list_parents, key=lambda x: x.fitness, reverse=False)[:n_parents]
        
        return best_parents
    
    def choose_survivors(self, n_survivors):
        survivors = sorted(self.population, key=lambda x: x.fitness, reverse=False)[:n_survivors]
        
        return survivors
    
if __name__ == "__main__":
    cti = cubesToImage()
    ga = GA()