from image_utils import *
import cube as cb
import random
import imageio
from multiprocessing import Pool

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
        self.target_image_gray_cubes = cubes_to_image(self.target_cubes, stroke_width=1, size_times=10, stroke_color=255)

        self.gas = np.zeros((self.n_cubes_x, self.n_cubes_x), dtype=object)
        self.bests = {}
        self.args = {}


    def process_cube(self, args):
        i, j = args
        ga = cb.cube_GA(self.target_cubes[i][j], **self.args)
        ga.run()
        gen = ga.generation
        best = ga.generation_info[gen]
        print(f"{i*self.n_cubes_x + j} - Generations {gen} - Fitness: {ga.get_best().fitness}")
        return i, j, best

    def run(self, args, filename=""):
        self.args = args
        self.bests = {}

        with Pool() as pool:
            results = pool.map(self.process_cube, [(i, j) for i in range(self.n_cubes_x) for j in range(self.n_cubes_x)])

        for i, j, best in results:
            if i not in self.bests:
                self.bests[i] = {}
            self.bests[i][j] = best

        if filename != "":
            import json
            with open('runs/' + filename + '.json', 'w') as outfile:
                json.dump(self.bests, outfile, indent=3)

    def run_not_parallel(self, args, filename=""):

        self.args = args

        for i in range(self.n_cubes_x):
            self.bests[i] = {}
            for j in range(self.n_cubes_x):
                self.gas[i][j] = cb.cube_GA(self.target_cubes[i][j], **args)
                self.gas[i][j].run()
                gen = self.gas[i][j].generation
                self.bests[i][j] = self.gas[i][j].generation_info[gen]
                print(str(i) + ","+ str(j) + "-" + " Generations " + str(self.gas[i][j].generation) + "- Fitness: " + str(self.gas[i][j].get_best().fitness))

        # write bests to json file
        if filename != "":
            import json
            with open('runs/' + filename + '.json', 'w') as outfile:
                json.dump(self.bests, outfile, indent=3)


    def best_image(self, stroke_width, size_multiply, border):
        final_cubes = np.zeros((self.n_cubes_x, self.n_cubes_x, 3, 3), dtype=np.uint8)

        for i in range(self.n_cubes_x):
            for j in range(self.n_cubes_x):
                final_cubes[i][j] = self.gas[i][j].get_best().image()

        return image_gray_to_image_cubes_color(cubes_to_image(final_cubes, stroke_width, size_multiply, border))

def best_image(self, stroke_width, size_multiply, border):
    final_cubes = np.zeros((self.n_cubes_x, self.n_cubes_x, 3, 3), dtype=np.uint8)

    for i in range(self.n_cubes_x):
        for j in range(self.n_cubes_x):
            final_cubes[i][j] = self.gas[i][j].get_best().image()

    return image_gray_to_image_cubes_color(cubes_to_image(final_cubes, stroke_width, size_multiply, border))

def animate_cube_movements(n_cubes_x, stroke_width, size_multiply, border, read_filename, write_filename, animation="parallel", duration=50):
    import json
    with open(read_filename) as json_file:
        data = json.load(json_file)

    cubes_history = []
    cubes_clone = []
    cubes_frames = []
    i = 0

    for _, row in data.items():
        for _, v in row.items():
            moves = v["best_moves"].split()
            face = v["best_face"]
            cubes_history.append(moves)
            cubes_clone.append(cb.Cube(np.zeros((3,3), dtype=np.uint8), fixedFace=face, move_history=[""]))
        i += 1

    frames = []
    i = 0
    while(max([len(x) for x in cubes_history]) > 0):
        cubes_images = []

        for z in range(len(cubes_clone)):

            if len(cubes_history[i]) > 0:
                cubes_clone[i].execute([cubes_history[i].pop(0)])

            cubes_images.append(cubes_clone[z].image())

        if len(cubes_history[i]) == 0:
            i += 1

        cubes_images = np.array(cubes_images).reshape((n_cubes_x, n_cubes_x, 3,3))
        frames.append(image_gray_to_image_cubes_color(cubes_to_image(np.array(cubes_images), stroke_width, size_multiply, border)))
        print("Frame " + str(len(frames)))

    frames = [imageio.imread(frame) if not isinstance(frame, np.ndarray) else frame for frame in frames]
    imageio.mimsave('images/gifs/' + write_filename + '.gif', frames, duration=duration, loop=0)

if __name__ == "__main__":
    cti = cubesToImage(n_cubes_x=5)
    args = {
        "population_size": 300,
        "mutation_rate": 0.9,
        "crossover_rate": 0.3,
        "max_generations": 10,
        "parent_pool_size": 10,
        "parents_number":  4
    }
