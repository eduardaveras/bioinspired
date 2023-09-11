import numpy as np
import time
from cube import Cube

class Rand:
    def __init__(self, __min__, __max__, size):
        self.values = []
        self.min = __min__
        self.max = __max__
        self.size = size
        self.seed = int(time.time())

        self.generate()

    def generate(self):
        np.random.seed(self.seed)
        for _ in range(self.size):
            value = np.random.randint(self.min, self.max)
            self.values.append(value)

    def pop (self):
        if len(self.values) == 0:
            self.generate()

        return self.values.pop(0)

class Rand_cube:
    def __init__(self, size):
        self.cubes = []
        self.size = size
        self.seed = int(time.time())

        self.generate()

    def generate(self):
        np.random.seed(self.seed)
        for _ in range(self.size):
            value = np.random.randint(0, 6, (3, 3))
            cube = Cube(value)
            self.cubes.append(cube)

    def pop(self):
        if len(self.cubes) == 0:
            self.generate()

        return self.cubes.pop(0)