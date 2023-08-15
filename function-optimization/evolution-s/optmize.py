import itertools
import evolution
import multiprocessing
import utils

# Defina os intervalos possíveis para cada hiperparâmetro
param_ranges = {
    "n_iterations": [50],
    "population_size": [400],
    "dimensions": [30],
    "number_of_parents": [50, 100, 300, 400],
    "function": ["rastrigin"],
    "learning_rate": [1.0, 2.0, 3.0],
    "global_learning_rate": [1.0, 2.0, 3.0],
    "selection_pressure": [2, 5, 7],
    "crossover": ["discrete", "intermediate"],
    "survivors_selection": ["plus_comma", "plus", "comma"],
    "parents_selection": ["best", "random"]
}

# Gere todas as combinações possíveis de hiperparâmetros
param_combinations = list(itertools.product(*param_ranges.values()))

# Função para executar o algoritmo com um conjunto de parâmetros e retornar o fitness
def run_evolution(params):
    evol = evolution.Evolution(**params)  # Use o módulo evolution aqui
    evol.run()
    evol.population.sort()
    return evol.population[0].fitness()

# Função para avaliar o desempenho de um conjunto de parâmetros com multiprocessamento
def evaluate_params(params):
    fitness = run_evolution(params)
    return fitness

if __name__ == "__main__":
    # Transforme cada tupla de parâmetros em um dicionário nomeado
    named_param_combinations = [dict(zip(param_ranges.keys(), param_values)) for param_values in param_combinations]

    total_combinations = len(named_param_combinations)
    print(f"Total combinations: {total_combinations}")

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use o número de núcleos da CPU

    # Utilize o tqdm para acompanhar o progresso
    from tqdm import tqdm
    results = list(tqdm(pool.imap(evaluate_params, named_param_combinations), total=total_combinations))

    pool.close()
    pool.join()

    best_result_index = results.index(max(results))
    best_params = named_param_combinations[best_result_index]
    best_fitness = results[best_result_index]

    print("Best Parameters:", best_params)
    print("Best Fitness:", best_fitness)
    # write the result in a file

    with open("result.txt", "w") as f:
        f.write(f"Best Parameters: {best_params}\n")
        f.write(f"Best Fitness: {best_fitness}\n")

# if __name__ == "__main__":
#     # Transforme cada tupla de parâmetros em um dicionário nomeado
#     named_param_combinations = [dict(zip(param_ranges.keys(), param_values)) for param_values in param_combinations]

#     pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())  # Use o número de núcleos da CPU
#     utils.blockPrint()
#     results = pool.map(evaluate_params, named_param_combinations)
#     utils.enablePrint()
#     print(max(results))
#     pool.close()
#     pool.join()

#     best_result_index = results.index(max(results))
#     best_params = named_param_combinations[best_result_index]
#     best_fitness = results[best_result_index]

#     print("Best Parameters:", best_params)
#     print("Best Fitness:", best_fitness)