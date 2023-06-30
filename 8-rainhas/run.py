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
        display("RUNNING: RUN " + str(r))
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

        # calculate mean fitness and std_dev
        fitness_total = 0
        sum_of_squared_differences = 0
        for p in alg.population:
            fitness_total += p.fitness
            sum_of_squared_differences += (p.fitness - fitness_total/alg.population_size)**2

        std_dev_fitness.append((sum_of_squared_differences/alg.population_size)**(1/2))
        mean_fitness.append(fitness_total/alg.population_size)

        # Mean of fitness in each iteration
        mean_per_iteration = []
        for fitness in alg.iteration_info:
            mean_per_iteration.append(sum(fitness)/len(fitness)) 

        runs.append({"iterations": alg.iterations, "mean_per_iteration": str(mean_per_iteration), "std_dev_fitness": std_dev_fitness[r], "mean_fitness": mean_fitness[r]})
        buff_time = "Time elapsed: " + str(round(sum(runs_times),2)) + " s"

        # output
        clear_output()
        display(buff_time)
        display(buff_2)

    test_name = "runs"
# print("Finished in " + str(round(sum(runs_times),2)) + " seconds")
    with open(filename + '.json', 'w') as outfile:
        json.dump(runs, outfile, indent=2)    

    # display(buff_2)
    clear_output()
    display(str(found_solution) + " solutions of " + str(n_runs) +  " were found in " + str(round(sum(runs_times),2)) + " seconds")

if __name__ == "__main__":
    run_(DEFAULT_ARGS)