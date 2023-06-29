import genetic as g
import board as eightqueens

__FILENAME = "runs" 
__NEW_INDIV = eightqueens.new_board
__ARGS = {"dna_size": 8, 
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
# Disable
def outputPrint(i):
    sys.stdout = open('output' + i, 'w')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

import json
import time
import sys, os

n_runs = 30 
runs = []
runs_times = []
mean_fitness = []
std_dev_fitness = []
found_solution = 0

iteration_number = []

for r in range(0, n_runs):
    start = time.time()
    enablePrint()
    print("  ->Running: run " + str(r))
    alg = g.Genetic(__NEW_INDIV, **__ARGS)
    blockPrint()
    alg.run()
    enablePrint()
    end = time.time()

    if alg.solution_was_found:
        _time = round(time.time() - start, 2)
        print(" Solution was found in " + str(_time) +  " seconds")
        runs_times.append(_time)
        found_solution += 1 
        iteration_number.append(alg.iterations)
    else: 
        print(" Solution was not found")
    
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

    runs.append({"run_" + str(r): {"iterations": alg.iterations, "mean_per_iteration": str(mean_per_iteration), "std_dev_fitness": std_dev_fitness[r], "mean_fitness": mean_fitness[r]}})
    print("[Total time: " + str(round(sum(runs_times),2)) + " seconds]")


test_name = "runs"
# print("Finished in " + str(round(sum(runs_times),2)) + " seconds")
with open(__FILENAME + '.json', 'w') as outfile:
    json.dump(runs, outfile, indent=2)    

enablePrint()
print(str(found_solution) + " solutions of " + str(n_runs) +  " were found in " + str(round(sum(runs_times),2)) + " seconds")