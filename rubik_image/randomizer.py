import numpy as np
import time
from cube import Cube

class Rand:
    def __init__(self, __min__, __max__, size):
        self.values = []
        self.size = size
        self.seed = int(time.time())
        self.__max__ = __max__
        self.__min__ = __min__

        self.generate()

    def generate(self):
        np.random.seed(self.seed)
        for _ in range(self.size):
            value = np.random.randint(self.__min__, self.__max__)
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
    
class Rand_index:
    def __init__(self, __min__, __max__, size):
        self.values = []
        self.seed = int(time.time())
        self.__min__ = __min__
        self.__max__ = __max__ + 1
        self.size = size
        
        self.generate()
        
    def generate(self):
        np.random.seed(self.seed)
        combinations = []
            
        for i in range(0, self.__max__):
            for j in range(0, self.__max__):
                combinations.append((i, j))
        
        for _ in range(self.size):
            np.random.shuffle(combinations)
            interval = np.random.randint(0, len(combinations), 2)
            tuples = combinations[min(interval):max(interval)]
            self.values.append(tuples)
        
    def pop (self):
        if len(self.values) == 0:
            self.generate()

        return self.values.pop(0)
    
class Rand_float:
    def __init__(self, size):
        self.values = []
        self.seed = int(time.time())
        self.size = size
        
        self.generate()
        
    def generate(self):
        np.random.seed(self.seed)
        for _ in range(self.size):
            self.values.append(np.random.random())
        
    def pop (self):
        if len(self.values) == 0:
            self.generate()

        return self.values.pop(0)
                    
            