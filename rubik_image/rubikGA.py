from randomizer import Rand, Rand_cube
from cube import Cube
from utils import load_image, image_to_gray_pixels, image_cubes
import numpy as np


class GA:
    def __init__(self, target_image_path, n_cubes_x=10, image_size=300, random_itens_size=10000):
        if image_size % n_cubes_x != 0:
            raise ValueError("Image size must be divisible by n_cubes")

        self.n_cubes_x = n_cubes_x
        self.n_cubes = n_cubes_x * n_cubes_x
        self.cube_size = int(image_size / (n_cubes_x * n_cubes_x))
        self.image_size = image_size
        self.population = []
        self.target_image = load_image(target_image_path)
        self.target_gray = image_to_gray_pixels(self.target_image, self.image_size, self.cube_size)
        self.target_cubes = image_cubes(self.target_image)
        self.random_cubes = Rand_cube(random_itens_size)
        self.random_ints = Rand(0, 6, random_itens_size)

    def init_population(self):
        for _ in range(self.n_cubes):
            new_individual = Individual()
            self.population.append(new_individual)

class Individual:
    def __init__(self, ga_instance, cubes=[]):
        self.cubes = cubes
        self.fitness = 0
        self.n_cubes = ga_instance.n_cubes
        self.random_cubes = ga_instance.random_cubes

        if cubes == []:
            self.random_init()

        self.evaluate_fitness()

    def evaluate_fitness(self):
        pass

    def random_init(self):
        for _ in range(self.n_cubes):
            cube_map = self.random_cubes.pop()
            self.cubes.append(Cube(cube_map))

    def to_image(self, cube_size, stroke_width):
        image = np.concatenate([cube.to_image(cube_size, stroke_width) for cube in self.cubes], axis=1)

def individual_to_image(individual):
    pass