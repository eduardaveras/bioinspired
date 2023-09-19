import ga
import cube
import numpy as np
import image_utils

path = "images/Nicolas_Cage_Deauville_2013.jpg"
cti = ga.cubesToImage(n_cubes_x=25, target_image_path=path)
args = {
    "population_size": 500,
    "mutation_rate": 0.9,
    "crossover_rate": 0.4,
    "max_generations": 2000,
    "parent_pool_size": 10,
    "parents_number":  4
}
cti.run(args, filename="nicolas_cage")
