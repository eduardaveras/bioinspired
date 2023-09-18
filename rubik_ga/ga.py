from image_utils import *
import cube as cb
import random
import imageio

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

        self.gas = np.zeros((self.n_cubes_x, self.n_cubes_x), dtype=object)

    def run(self, args):
        for i in range(self.n_cubes_x):
            for j in range(self.n_cubes_x):
                self.gas[i][j] = cube.cube_GA(self.target_cubes[i][j], **args)
                self.gas[i][j].run()
                print("" + str(i) + ","+ str(j) + " " + "Generations" + str(self.gas[i][j].generation) + "Fitness:" + str(self.gas[i][j].get_best().fitness))

    def best_image(self, stroke_width, size_multiply, border):
        final_cubes = np.zeros((self.n_cubes_x, self.n_cubes_x, 3, 3), dtype=np.uint8)

        for i in range(self.n_cubes_x):
            for j in range(self.n_cubes_x):
                final_cubes[i][j] = self.gas[i][j].get_best().image()

        return image_gray_to_image_cubes_color(cubes_to_image(final_cubes, stroke_width, size_multiply, border))

def cube_to_animation(target_cube, moves, filename="cube_moves", fixedFace=None, save=False):
    cube = cb.Cube(np.zeros((3, 3, 3)), move_history=["x"], fixedFace=fixedFace)
    frames = []
    initial_image = image_gray_to_image_cubes_color(cube.image(150))
    frames.append(initial_image)

    for move in moves:
        cube.execute([move])
        img = image_gray_to_image_cubes_color(cube.image(150))
        frames.append(img)

    frames = [imageio.imread(frame) if not isinstance(frame, np.ndarray) else frame for frame in frames]
    if save:
        imageio.mimsave('images/' + filename + '.gif', frames, duration=0.5, loop=0)


if __name__ == "__main__":
    cti = cubesToImage()
    args = {
        "population_size": 1000,
        "mutation_rate": 0.9,
        "crossover_rate": 0.3,
        "max_generations": 1000,
        "parent_pool_size": 10,
        "parents_number":  4
    }

    gas = [cube_GA(cti.target_cubes[i], **args) for i in range(cti.n_cubes_x**2)]
