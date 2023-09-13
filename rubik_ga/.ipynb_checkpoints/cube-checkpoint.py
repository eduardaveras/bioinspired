import lib
import numpy as np
import random
from constants import *
from image_utils import corresponding_gray

class Cube(lib.Cube):
    def __init__(self, target_gray_face):
        super().__init__()
        self.move_history = []
        self.target_gray_face = target_gray_face
        self.fitness = 0

    def front_face(self, gray=False):
        front_face = self.faces[FRONT]

        if gray:
            gray_face = np.zeros((3, 3), dtype=np.uint8)
            for i in range(3):
                for j in range(3):
                    gray_face[i, j] = corresponding_gray(front_face[i, j])
        return gray_face

        return front_face

    def random_scramble(self, length=20):
        return [random.choice(SINGLE_MOVES) for _ in range(length)]
        
    def evaluate_fitness(self):
        gray_face = self.front_face(gray=True)
        self.fitness = np.sum(np.abs(gray_face.astype(int) - self.front_face(True).astype(int)))