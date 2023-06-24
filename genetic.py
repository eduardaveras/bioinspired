import random
from board import new_board, Board
"""

Primeira parte:
Representação (genótipo): string de bits
Recombinação: “cut-and-crossfill” crossover
Probabilidade de Recombinação: 90%
Mutação: troca de genes
Probabilidade de Mutação: 40%
Seleção de pais: ranking - Melhor de 2 de 5 escolhidos aleatoriamente
Seleção de sobreviventes: substituição do pior
Tamanho da população: 100
Número de filhos gerados: 2
Inicialização: aleatória
Condição de término: Encontrar a solução, ou 10.000 avaliações de fitness
Fitness?


"""
GENESET = "01"
RECOMBINATION_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.4
POPULATION_SIZE = 100
CHILDREN_NUMBER = 2

class Genetic:
    """ 

    """
    def __init__(self, dna_size, new_indiv_func, population_size=100, 
                 gene_set="01", recombination_probability=0.9, 
                 mutation_probability=0.4, children_number=2):


        self.dna_size = dna_size
        self.population_size = population_size
        self.geneset = gene_set
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.children_number = children_number
        self.new_indiv_func = new_indiv_func

        self.population = self.init_population()
    # def run(self):
    #     self.population = self.init_population(new_board)
        # pass
    
    def init_population(self):
        population = []  
        for i in range(self.population_size):
            population.append(self.new_indiv_func(self.dna_size))
        
        return population

    def crossover_cutandfill(parent1, parent2):
        cut = random.randint(0, len(parent1))
        child1 = parent1[:cut] + parent2[cut:]
        child2 = parent2[:cut] + parent1[cut:]
        return child1, child2


    def single_mutation(indiv, geneSet):
        index = random.randrange(0, len(indiv))
        indiv_mutated = list(indiv)
        new_gene, alternate = random.sample(geneSet, 2)
        indiv_mutated[index] = alternate if new_gene == indiv_mutated[index] else new_gene
        return ''.join(indiv_mutated)   

    def double_mutation(indiv):
        indices= random.sample(range(len(indiv)), 2)
        indiv_mutated = list(indiv)
        temp = indiv_mutated[indices[0]]
        indiv_mutated[indices[0]] = indiv_mutated[indices[1]]
        indiv_mutated[indices[1]] = temp
        return ''.join(indiv_mutated)

    def parent_tournament(self, population, choices_size):
        list_parents = random.sample(population, choices_size)
        best_parents = sorted(list_parents, key=lambda indiv: indiv.fitness, reverse=True)

        return best_parents
            


    def get_best(get_fitness, target_len, optimal_fitness, geneSet, display, mutate_function):
        if not optimal_fitness > best_parent.Fitness:
            return best_parent
        
        while True:
            child = mutate_function(best_parent, geneSet, get_fitness)
            if not child.Fitness > best_parent.Fitness:
                continue
            display(child)
            
            if not optimal_fitness > child.Fitness:
                return child
            best_parent = child


if __name__ == '__main__':
    g = Genetic(8, new_board, population_size=3)
    for indiv in g.population:
        # indiv.show()
        print(indiv.fitness)

    print()
    for i in sorted(g.population, key=lambda indiv: indiv.fitness, reverse=True):
        print(i.fitness)

    

