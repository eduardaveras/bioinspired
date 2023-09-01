import numpy as np
import time

class Rand:
    def __init__(self, __min__, __max__, size):
        self.values = []
        self.min = __min__
        self.max = __max__
        self.size = size
        self.seed = int(time.time())

        self.generate()

    def generate(self):
        # Generate a list of random numbers
        np.random.seed(self.seed)
        self.values = np.random.randint(self.min, self.max, self.size)

    def pop (self):
        if len(self.values) == 0:
            self.generate()

        return self.values.pop(0)

class Rand_cube:
    def __init__(self, size):
        self.values = []
        self.size = size
        self.seed = int(time.time())

        self.generate()

    def generate(self):
        np.random.seed(self.seed)

        for _ in range(self.size):

            value = np.random.randint(0, 6, (3, 3))

            self.values.append(value)

    def pop(self):
        if len(self.values) == 0:
            self.generate()

        return self.values.pop(0)