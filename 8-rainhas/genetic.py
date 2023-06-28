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

class Genetic:
    def __init__(self, dna_size, new_indiv_func, population_size=100, 
                 gene_set="01", recombination_probability=0.9, 
                 mutation_probability=0.4, children_number=2,
                 max_iterations=10000
                 ):


        self.solution_was_found = False
        self.dna_size = dna_size
        self.population_size = population_size
        self.gene_set = gene_set
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.children_number = children_number
        self.new_indiv_func = new_indiv_func
        self.population = []
        self.max_iterations = max_iterations
        self.iterations = 0
        self.iteration_info = []
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability

    def run(self):
        """
        begin
            initialise population with random candidate solution;
            evaluate each candidate;
            repeat until (termination condition is satisfied) do
                1 select parents;
                2 recombine pairs of parents
                3 mutate the resulting offspring;
                4 evaluate new candidates;
                5 select individuals for the next generation;
            od
        end
        """
        # initilize population
        self.population = self.init_population()

        while not self.finish_condition(self.population):
            self.iterations += 1
            children = []
            print(f"\n{self.iterations} iteration ---------\n")
            print(f"Population fitnesses: {[i.fitness for i in self.population]}", end='\n', sep=',')

            parents = self.parent_tournament(self.population, 5, 2)

            if random.choices([True, False], weights=[self.recombination_probability, 1-self.recombination_probability], k=1)[0]:
                print(f"Recombining parents => ", end=' ')
                child1, child2 = self.crossover_cutandfill(parents[0], parents[1])
                children = [child1, child2]
                print(f"Children from crossover: {[(i.dna, i.fitness) for i in children]}", end='\n', sep=',')
            else:
                children = parents
                print(f"Same as the parents: {[i.fitness for i in children]}" ,end='\n', sep=' ')

            for indiv in self.population:
                if indiv not in parents and \
                    random.choices([True, False], weights=[self.mutation_probability, 1-self.mutation_probability], k=1)[0]:
                        indiv_mutated = self.new_indiv_func(self.dna_size, dna=self.single_mutation(indiv, self.gene_set))
                        self.switch_indiv(indiv, indiv_mutated)

            self.population = self.choose_survivor(self.population + children, self.population_size)
            self.population = sorted(self.population, key=lambda indiv: indiv.fitness, reverse=True)
            self.iteration_info.append([i.fitness for i in self.population])

            print("---------------------------")

        return min(self.population, key=lambda indiv: indiv.fitness)
                
    def switch_indiv(self, indiv1, indiv2):
        self.population.remove(indiv1)
        self.population.append(indiv2)


    def finish_condition(self, population):
        best_indiv = max(population, key=lambda indiv: indiv.fitness)

        if best_indiv.fitness == Board(self.dna_size).get_max_fitness():
            best_indiv.show()
            print(f"Found the solution, ending...")
            self.solution_was_found = True
            return True
        
        if self.iterations == self.max_iterations:
            print("Max iteration number was reached")
            return True

        return False

       
    def init_population(self):
        population = []  
        for i in range(self.population_size):
            population.append(self.new_indiv_func(self.dna_size))
        
        return population
    
    def crossover_cutandfill(self, parent1, parent2):
        cut = random.randint(0, len(parent1.dna)-1)
        #print(f"Cut point:  {cut}", end='\n')
        dna1 = parent1.dna[:cut] + parent2.dna[cut:]
        dna2 = parent2.dna[:cut] + parent1.dna[cut:]
        
        child1 = Board(self.dna_size, dna=dna1)
        child2 = Board(self.dna_size, dna=dna2)

        return child1, child2

    def single_mutation(self, indiv, geneSet):
        index = random.randrange(0, len(indiv.dna))               
        indiv_mutated = list(indiv.dna)
        new_gene, alternate = random.sample(geneSet, 2)
        indiv_mutated[index] = alternate if new_gene == indiv_mutated[index] else new_gene
        dna_mutated = ''.join(indiv_mutated)   

        return dna_mutated

    def double_mutation(self, indiv):
        indices= random.sample(range(len(indiv)), 2)
        indiv_mutated = list(indiv)
        temp = indiv_mutated[indices[0]]
        indiv_mutated[indices[0]] = indiv_mutated[indices[1]]
        indiv_mutated[indices[1]] = temp
        
        return ''.join(indiv_mutated)

    def parent_tournament(self, population, choices_size, return_size):
        list_parents = random.sample(population, choices_size)
        best_parents = sorted(list_parents, key=lambda indiv: indiv.fitness, reverse=True)
        print(f"Selecting parents: {[i.fitness for i in best_parents]}", end='\n', sep=',')
        # Best parents:
        print(f"Chosen parents: {[i.fitness for i in best_parents][:return_size]}", end='\n', sep=',')

        return best_parents[:return_size]

    def parent_spinwheel(self, population, return_size):
        fitness_sum = sum([indiv.fitness for indiv in population])
        fitnesses = [indiv.fitness/fitness_sum for indiv in population]

        return random.choices(population, weights=fitnesses, k=return_size) 

    def choose_survivor(self, population, return_size):
        return sorted(population, key=lambda indiv: indiv.fitness, reverse=True)[:return_size]
    

# if __name__ == '__main__':
    # g = Genetic(8, new_board)
    
    # Test for parent_toournament 
    # p = g.parent_tournament(g.population, 5, 2)
    # print(f"Parents= {[i.fitness for i in p]}", end='\n', sep=',')
    
    # Test for choose_survivor
    # s = g.choose_survivor(g.population, 5)
    # print(f"Survivors= {[i.fitness for i in s]}", end='\n', sep=',')               

    # Test for single_mutation
    #indiv = g.population[0]
    #indiv_dna = g.single_mutation(indiv, "01")
    #indiv_mutated = new_board(4, dna=indiv_dna)       
    #indiv_mutated.show()
    
    #Test for board.show()
    #for indiv in g.population:
    #    indiv.show()
    #    print("max fitness:", end=' ')
    #    print(indiv.get_max_fitness(), end='\n')
    
    # g.run()
    # print(f"Population= {[i.fitness for i in g.population]}", end='\n', sep=',')

    
