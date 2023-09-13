import lib
import numpy as np
import random
from constants import *
from image_utils import corresponding_gray, corresponding_color, resize_pixeled, gray_nearest

class Cube(lib.Cube):
    def __init__(self, target_gray_face, __random__=True):
        super().__init__()
        self.move_history = []
        self.target_gray_face = target_gray_face
        self.fitness = 0
        self.random_init(__random__)

        self.evaluate_fitness() 

    def image(self, resize=0):
        __image__ = np.zeros((3, 3, 3), np.uint8)
        __front_face__ = self.front_face()

        for i in range(3):
            for j in range(3):
                __image__[i, j] = corresponding_color(__front_face__[i,j]) 

        if resize != 0:
            return resize_pixeled(__image__, resize)
        
        return __image__

    def front_face(self, gray=False):
        front_face = self.faces[FRONT]

        if gray:
            gray_face = np.zeros((3, 3), dtype=np.uint8)
            for i in range(3):
                for j in range(3):
                    gray_face[i, j] = corresponding_gray(front_face[i, j])
            return gray_face

        return front_face

    def random_init(self, __random__, length=20):
        if __random__:
            scramble = [random.choice(SINGLE_MOVES) for _ in range(length)]
            self.execute(scramble)
        else:
            self.execute(["x'"])
        
    def evaluate_fitness(self):
        gray_face = self.front_face(gray=True)
        target_gray = self.target_gray_face
        self.fitness = (np.sum(np.abs(gray_face.astype(int) - target_gray.astype(int)))) / (2034)

