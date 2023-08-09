## PARAMETROS PARA CADA FUNÇÃO:

--------------- Estratégia evolutiva ---------------

1. Ackley:

    population_size = 300,
    number_of_parents = 300,
    crossover_rate = 1,
    learning_rate = 1.1,
    global_learning_rate = 1.1,
    crossover = "discrete",
    survivors_selection = "plus",
    parents_selection = "best"
    selection_pressure = 7

2. Rastrigin:

    number_of_parents = 200,
    crossover_rate = 1,
    learning_rate = 2,
    global_learning_rate = 2,
    crossover = "intermediate",
    survivors_selection = "plus",
    parents_selection = "best"

3. Rosenbrock:

    number_of_parents = 200,
    crossover_rate = 1,
    learning_rate = 2, 
    global_learning_rate = 2, 
    crossover = "intermediate", 
    survivors_selection = "plus", 
    parents_selection = "best"

4. Schwefel:

    number_of_parents = 200,
    crossover_rate = 1,
    learning_rate = 2, 
    global_learning_rate = 2, 
    crossover = "intermediate", 
    survivors_selection = "plus", 
    parents_selection = "best"


--------------- Algoritmo genético ---------------

1. Ackley:
    population_size=100, 
    pair_children_size=4, 
    recombination_probability=0.9,
    index_mutation_probability=0.5, 
    index_mutation_rate=0.99998,
    mutation_method="gaussian", 
    mutation_probability=0.1,
    parent_method="tournament", 
    survivor_method="best",