import genetic as g
import board as eightqueens
from utils import *
from IPython.display import clear_output, display

__FILENAME = "teste"
__NEW_INDIV = eightqueens.new_board
DEFAULT_ARGS = {"dna_size": 8,
                "population_size": 100,
                "gene_set": "01",
                "max_iterations": 10000,
                "parent_method": "tournament",
                "survivor_method": "best",
                "recombination_method": "cutandfill",
                "recombination_probability": 0.9,
                "mutation_method": "single",
                "mutation_probability": 0.4
        }

# ---------------------------------

def run_(args, filename=__FILENAME, n_runs=30):
    import json
    import time
    import numpy as np

    runs = []
    runs_times = []
    mean_fitness = []
    std_dev_fitness = []
    found_solution = 0
    iteration_number = []

    buff_2 = [""]
    _time = 0.
    for r in range(0, n_runs):
        # output
        #CLock
        start = time.time()

        # Alg running
        outputPrint(filename)
        alg = g.Genetic(__NEW_INDIV, **args)
        alg.run()
        enablePrint()

        end = time.time()
        _time = round(time.time() - start, 2)

        if alg.solution_was_found:
            _time = round(time.time() - start, 2)
            buff_2 = "RUN " + str(r) + " STATUS: FOUND       in " + str(_time) + "s and " + str(alg.iterations) + " iterations"
            found_solution += 1
            iteration_number.append(alg.iterations)
        else:
            buff_2 = "RUN " + str(r) + " STATUS: NOT FOUND   in " + str(_time) + "s and " + str(alg.iterations) + " iterations"

        runs_times.append(_time)
        buff_time = "Time elapsed: " + str(round(sum(runs_times),2)) + " s"

        # calculate mean fitness and std_dev
        fitness = [p.fitness for p in alg.population]
        _mean = np.mean(fitness)
        _std = np.std(fitness)
        std_dev_fitness.append(_std)
        mean_fitness.append(_mean)

        # Mean of fitness in each iteration
        mean_per_iteration = []
        std_per_iteration = []
        best_per_iteration = []
        for fitness in alg.iteration_info:
            _best = int(np.max(fitness))
            __mean = np.mean(fitness)
            mean_per_iteration.append(__mean)
            best_per_iteration.append(_best)

        runs.append({"found_solution": alg.solution_was_found, "iterations": alg.iterations, "mean_per_iteration": mean_per_iteration, "best_per_iteration": best_per_iteration,  "std": _std, "mean": _mean, "last_population_fitness": fitness})

        # output
        clear_output()
        display("Solutions: " + str(found_solution) + " of " + str(r) + ", " + buff_time)
        display(buff_2)

    test_name = "runs"
# print("Finished in " + str(round(sum(runs_times),2)) + " seconds")
    with open(filename + '.json', 'w') as outfile:
        json.dump(runs, outfile, indent=2)

    # display(buff_2)
    clear_output()
    display(str(found_solution) + " solutions of " + str(n_runs) +  " were found in " + str(round(sum(runs_times),2)) + " seconds")

if __name__ == "__main__":
    args = DEFAULT_ARGS
    args["max_iterations"] = -1
    run_(args)
