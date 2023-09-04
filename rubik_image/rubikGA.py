from randomizer import Rand, Rand_cube
from cube import Cube
from utils import load_image, image_to_gray_pixels, gray_image_to_cubes, cubes_to_image
import numpy as np


class GA:
    def __init__(self, target_image_path="./images/monalisa.jpg", n_cubes_x=20, image_size=300, random_itens_size=10000):
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
        self.random_ints = Rand(0, 6, random_itens_size)

    def init_population(self):
        for _ in range(self.n_cubes):
            new_individual = Individual(self)
            self.population.append(new_individual)

class Individual:
    def __init__(self, ga_instance, cubes=None):
        self.fitness = 0
        self.n_cubes_x = ga_instance.n_cubes_x
        self.random_cubes = ga_instance.random_cubes

        if cubes == None:
            self.random_init()

        elif type(cubes) is not np.ndarray:
            raise TypeError("cubes initializer must be a numpy array")

        self.evaluate_fitness(ga_instance.target_cubes)

    def evaluate_fitness(self, target_cubes):
        for i in range(0, self.n_cubes_x):
            for j in range(0, self.n_cubes_x):
                self.fitness += np.sum(np.abs(self.cubes[i][j].gray_map - target_cubes[i][j]))

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