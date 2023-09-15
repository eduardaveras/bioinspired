from constants import *
import numpy as np

def chance(probability):
    if np.random.random() < probability:
        return True

    return False

def random_single_move(probability=0.5):
    index = np.random.randint(0, len(SINGLE_MOVES))

    if chance(probability):
        return SINGLE_MOVES[index]

    return []

def random_orientation(probability=0.5):
    index = np.random.randint(0, len(ORIENTATIONS))

    if chance(probability):
        return ORIENTATIONS[index]

    return []

def random_rotation(probability=0.5):

    index = np.random.randint(0, len(FULL_ROTATIONS))

    if chance(probability):
        return FULL_ROTATIONS[index]

    return []

def random_permutation(probability=0.5):
    index = np.random.randint(0, len(PERMUTATIONS))

    if chance(probability):
        return PERMUTATIONS[index]

    return []

def random_scramble(rotation_probability=0.5, orientation_probability=0.5, permutation_probability=0.5, single_move_probability=0.5):
    return np.concatenate((random_rotation(rotation_probability), random_orientation(orientation_probability), random_permutation(permutation_probability), random_single_move(single_move_probability)), axis=None)