from matplotlib  import pyplot as plt
import genetic as g
import board as eightqueens
import json
import time
import sys, os

# Disable
def outputPrint(i):
    sys.stdout = open('output' + i, 'w')

def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

n_runs = 30 
new_indiv_func = eightqueens.new_board

runs = []
runs_times = []
mean_fitness = []
std_dev_fitness = []
found_solution = 0

iteration_number = []

for r in range(0, n_runs):
    start = time.time()
    enablePrint()
    print("Running: run " + str(r))
    alg = g.Genetic(8, new_indiv_func)
    blockPrint()
    alg.run()
    enablePrint()
    end = time.time()

    if alg.solution_was_found:
        print("Solution was found in " + str(time.time() - start) +  " seconds")
        runs_times.append(time.time() - start)
        found_solution += 1 
        iteration_number.append(alg.iterations)
    
    # calculate mean fitness
    fitness_total = 0
    sum_of_squared_differences = 0
    for p in alg.population:
        fitness_total += p.fitness
        sum_of_squared_differences += (p.fitness - fitness_total/alg.population_size)**2

    std_dev_fitness.append((sum_of_squared_differences/alg.population_size)**(1/2))
    mean_fitness.append(fitness_total/alg.population_size)
    
    # write alg.iterations, std_dev_fitness, mean_fitness to a json
    runs.append({"run_" + str(r): {"iterations": alg.iterations, "std_dev_fitness": std_dev_fitness[r], "mean_fitness": mean_fitness[r]}})

# json_runs = json.dumps(runs, indent=4)
# write runs in a json file
print("Finished in " + str(sum(runs_times)) + " seconds")
with open('runs.json', 'w') as outfile:
    json.dump(runs, outfile, indent=4)    

enablePrint()
print(found_solution)