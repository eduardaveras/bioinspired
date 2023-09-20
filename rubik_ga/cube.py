import lib
import numpy as np
from constants import *
from image_utils import corresponding_gray, corresponding_color, resize_pixeled, gray_nearest
from cube_utils import *
from functools import total_ordering
import cv2
import json

@total_ordering
class Cube(lib.Cube):
    def __init__(self, target_gray_face, move_history=[], fixedFace=None):
        super().__init__()
        self.move_history = move_history
        self.target_gray_face = target_gray_face
        self.fitness = 2
        self.fitness_face = FRONT
        self.output = ""
        self.fixedFace = fixedFace

        self.init(move_history)

    def __eq__(self, other):
        return self.fitness == other.fitness

    def __lt__(self, other):
        return self.fitness < other.fitness

    def best_face(self):
        if self.fitness_face is not None and isinstance(self.fitness_face, str):
            return self.faces[self.fitness_face]
        
        raise Exception("Fitness face is not a string")

    def add_exec_eval(self, move):
        self.move_history = np.concatenate((self.move_history, move))
        self.execute(move)
        self.evaluate_fitness()

        if self.fixedFace is not None:
            self.fitness_face = self.fixedFace

    def image(self, resize=0, stroke=False, strokeColor=0):
        if self.fitness_face is None or not isinstance(self.fitness_face, str):
            raise Exception("Fitness face is not a string")

        __image__ = np.zeros((3,3) , np.uint8)
        __front_face__ = self.faces[self.fitness_face]

        for i in range(3):
            for j in range(3):
                __image__[i, j] = corresponding_gray(__front_face__[i,j])

        if resize != 0:
            img = resize_pixeled(__image__, resize)
            if stroke:
                return cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=strokeColor)

            return resize_pixeled(__image__, resize)

        return __image__

    def gray_faces(self):

        gray_faces = {}
        for face, value in self.faces.items():
            gray_face = np.zeros((3, 3), dtype=np.uint8)
            for i in range(3):
                for j in range(3):
                    gray_face[i, j] = corresponding_gray(value[i, j])

            gray_faces[face] = gray_face

        return gray_faces

    def init(self, move_history, length=3):
        self.execute(["x'"]) # Começa com a face branca para cima
        if len(move_history) == 0:
            scramble = [np.random.choice(SINGLE_MOVES) for _ in range(length)]
            move_history = scramble

        self.add_exec_eval(move_history)

    def evaluate_fitness(self):
        gray_faces = self.gray_faces()
        target_gray = self.target_gray_face

        best = (None, 2)
        for identification, face in gray_faces.items():
            fitness = (np.sum(np.abs(face.astype(int) - target_gray.astype(int)))) / (2034)

            if fitness < best[1]:
                best = (identification, fitness)

        self.fitness_face, self.fitness = best


    def mutate(self):
        scramble = random_scramble(orientation_probability=0.2,
                                   permutation_probability=0.4,
                                   single_move_probability=0.8,
                                   rotation_probability=0.2
                                   )
        self.add_exec_eval(scramble)

    def export(self):
        return json.dumps({"fitness": self.fitness, "move_history": ' '.join(self.move_history.tolist())})

    def crossover_with(self, other):
        _cut_point = np.random.randint(0, len(self.move_history))
        other_cut_point = int(_cut_point / len(self.move_history) * len(other.move_history))

        _dna = (self.move_history[_cut_point:], self.move_history[:_cut_point])
        other_dna = (other.move_history[other_cut_point:], other.move_history[:other_cut_point])

        children = (Cube(self.target_gray_face, move_history=np.concatenate((_dna[0], other_dna[1]))),
                    Cube(self.target_gray_face, move_history=np.concatenate((_dna[1], other_dna[0]))))

        return children

class cube_GA:
    def __init__(self, target_gray_face, population_size=1000, mutation_rate=0.9,
                 crossover_rate=0.3, max_generations=500,
                 parent_pool_size=10, parents_number=4):

        self.target_gray_face = target_gray_face
        self.population_size = 100
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_generations = max_generations
        self.parent_pool_size = parent_pool_size
        self.parents_number = parents_number

        # Non-set
        self.population = []
        self.parents = []
        self.children = []
        self.generation = 0
        self.best_cube = None
        self.generation_info = {}
        self.output = []

    def get_best(self, __list__=None):
        if __list__ is None:
            __list__ = self.population

        return min(__list__)

    def get_bests(self, n=1, __list__=None):
        if __list__ is None:
            __list__ = self.population

        return sorted(__list__, reverse=False)[:n]

    def get_bests_fitness(self, n=1, __list__=None):
        if __list__ is None:
            __list__ = self.population

        list_ = sorted(__list__, reverse=False)[:n]
        return [cube.fitness for cube in list_]

    def get_worst(self, __list__=None):
        if __list__ is None:
            __list__ = self.population

        return max(__list__)

    def get_average_fitness(self):
        return np.mean([cube.fitness for cube in self.population])

    def init_info(self):
        self.generation_info[self.generation] = {}

    def append_info(self, key=None, info=None):
        _info = self.generation_info[self.generation]

        if key is None:
            _info["best_fitness"] = self.get_best().fitness
            _info["best_cube"] = (str(self.get_best().best_face().tolist()))
            _info["best_moves"] = ' '.join(self.get_best().move_history)
            _info["best_face"] = self.get_best().fitness_face
            _info["population"] = self.get_bests_fitness(n=len(self.population))
            _info["parents"] = self.get_bests_fitness(__list__=self.parents)
            _info["children"] = self.get_bests_fitness(__list__=self.children)
            _info["mean_fitness"] = np.mean(_info["population"])
            _info["std_fitness"] = np.std(_info["population"])
            _info["population"] = str(_info["population"])

            self.output += f"Generation: {self.generation}\n" + \
                           f"Best fitness: {_info['best_fitness']}\n" + \
                           f"Best moves: {_info['best_moves']}\n" + \
                           f"Parents: {_info['parents']}\n" + \
                           f"Children: {_info['children']}\n" + \
                           f"-------------------\n"

        else:
            _info[key] = info
            self.output += f"{key}: {info}\n"


    def log(self):
        print(self.output)
        self.output = ""

    def init_population(self):
        self.population = [Cube(self.target_gray_face) for _ in range(self.population_size)]

    def get_survivors(self):
        self.population = self.get_bests(__list__=self.population + self.children, n=self.population_size)

    def end_condition(self):
        if self.max_generations == self.generation:
            return True

        if self.get_best().fitness == 0 and self.generation > 0:
            return True

        return False

    def tournament(self):
        indexes = np.random.choice(len(self.population), self.parent_pool_size, replace=False)
        parent_pool = [self.population[i] for i in indexes]

        self.append_info(key="parents_pool", info=str([parent.fitness for parent in parent_pool]))
        return self.get_bests(n=self.parents_number, __list__=parent_pool)

    def generate_children(self, crossover_rate=0.5):
        indexes = np.arange(len(self.parents))
        np.random.shuffle(indexes)
        shuffled_parents = [self.parents[i] for i in indexes]

        parent_pairs = [(shuffled_parents[i], shuffled_parents[i + 1]) for i in range(0, len(shuffled_parents), 2)]

        children = []
        for pair in parent_pairs:
            if chance(crossover_rate):
                child_pair = pair[0].crossover_with(pair[1])
                children.append(child_pair[0])
                children.append(child_pair[1])
            else:
                children.append(pair[0])
                children.append(pair[1])

        self.append_info(key="Children before mutation", info=[child.fitness for child in children])
        return children

    def run(self, verbose=False, animate=False):
        self.init_population()

        while not self.end_condition():
            self.generation += 1
            self.init_info()
            self.parents = self.tournament()
            self.children = self.generate_children()

            for child in self.children:
                if chance(self.mutation_rate):
                    child.mutate()

            self.append_info()
            self.get_survivors()
            if verbose:
                self.log()

        return self.generation_info

if __name__ == "__main__":
    test_target = np.array([[125, 125, 125],
                            [ 76,  76,  29],
                            [ 76,  29,  29]], dtype=np.uint8)
    
    ga = cube_GA()
    ga.run(verbose=True, animate=False)
