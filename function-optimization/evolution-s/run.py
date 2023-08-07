from evolution import Evolution
from chromossome import Chromossome

import random

e = Evolution()

def run(num_iterations):
    fitness = []
    iteration = 0
    
    # Inicialize o indivíduo x com valores aleatórios
    population = e.init_population()
    
    # Avalie a aptidão (ou fitness) de cada indivíduo
    for i in range(e.population_size):
        fitness.append(population[i].fitness())
        
    # defina o tamanho do passo
    
    # critere de parada
    while(iteration < num_iterations):
        index = random.randint(0, e.population_size - 1)
        new_indiv = population[index].copy()
        new_indiv.mutate()
        if (new_indiv.fitness() > population[index].fitness()):
            population[index] = new_indiv
            
        if (new_indiv.fitness() == 0):
            return new_indiv.fitness()
        
        iteration += 1
        
        
        
        