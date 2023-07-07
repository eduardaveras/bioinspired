import random
from board import new_board, Board, board_to_binary
from datetime import datetime
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
    def __init__(self, new_indiv_func, dna_size=8, population_size=100, max_iterations=10000,
                 genotipe_size=3, gene_set="01", chrildren_size=2,
                 recombination_method="cutandfill", recombination_probability=0.9,
                 mutation_method="single", mutation_probability=0.4,
                 parent_method="tournament", survivor_method="best",
                 ):


        self.solution_was_found = False
        self.dna_size = dna_size
        self.genotipe_size = genotipe_size
        self.population_size = population_size
        self.gene_set = gene_set
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability
        self.new_indiv_func = new_indiv_func
        self.population = []
        self.max_iterations = max_iterations
        self.iterations = 0
        self.iteration_info = []
        self.recombination_probability = recombination_probability
        self.mutation_probability = mutation_probability

        self.parent_method = parent_method
        self.survivor_method = survivor_method
        self.recombination_method = recombination_method
        self.mutation_method = mutation_method

    def run(self):
        random.seed(datetime.now().timestamp())
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
            parents = []
            print(f"\n{self.iterations} iteration ---------\n")
            print(f"Population fitnesses: {[i.fitness for i in self.population]}", end='\n', sep=',')

            # Selection of parents
            if self.parent_method == "tournament":
                parents = self.parent_tournament(self.population, 5, 2)
            elif self.parent_method == "spinwheel":
                parents = self.parent_spinwheel(self.population, 2)
            else:
                raise Exception("Invalid parent method")

            # Recombination
            if random.choices([True, False], weights=[self.recombination_probability, 1-self.recombination_probability], k=1)[0]:
                print(f"Recombining parents => ", end=' ')

                # Recombination occurs
                child1, child2 = None, None
                if self.recombination_method == "cutandfill":
                    child1, child2 = self.crossover_cutandfill(parents[0], parents[1])
                else:
                    raise Exception("Invalid recombinatio method")

                children.append(child1)
                children.append(child2)

                print(f"Children from crossover: {[(i.dna, i.fitness) for i in children]}", end='\n', sep=',')
            else:
                children.append(parents[0])
                children.append(parents[1])
                print(f"Same as the parents: {[i.fitness for i in children]}" ,end='\n', sep=' ')

            for indiv in children:
                self.population.append(indiv)
                if random.choices([True, False], weights=[self.mutation_probability, 1-self.mutation_probability], k=1)[0]:

                    # Mutating
                    dna_mutated = None
                    if self.mutation_method == "single":
                        dna_mutated = self.single_mutation(indiv)
                    elif self.mutation_method == "double":
                        dna_mutated = self.double_mutation(indiv)
                    elif self.mutation_method == "singledouble":
                        dna_mutated = self.double_mutation(indiv)
                    elif self.mutation_method == "pertube":
                        dna_mutated = self.pertube_mutation(indiv)
                    else :
                        raise Exception("Invalid mutation method")

                    indiv_mutated = self.new_indiv_func(self.dna_size, dna=dna_mutated)
                    print("Mutaded: " + str((indiv_mutated.dna, indiv_mutated.fitness)))
                    self.switch_indiv(indiv, indiv_mutated)

            # Survivor selection
            if self.survivor_method == "best":
                self.choose_survivor(self.population_size)
            elif self.survivor_method == "generational":
                self.population.remove(parents[0])
                self.population.remove(parents[1])
            else:
                raise Exception("Invalid survivor method")
            # elif self.survivor_method == "random":

            # self.population = sorted(self.population, key=lambda indiv: indiv.fitness, reverse=True)
            self.iteration_info.append([i.fitness for i in self.population])

            print("---------------------------")

    def switch_indiv(self, indiv1, indiv2):
        self.population.remove(indiv1)
        self.population.append(indiv2)

    def gene_block(self, indiv):
        return [indiv.dna[i:i+self.genotipe_size] for i in range(0, len(indiv.dna), self.genotipe_size)]

    def finish_condition(self, population):
        solutions = [indiv.isSolution for indiv in population]
        print("We found " + str(solutions.count(True)) + " solutions")

        if True in solutions and self.max_iterations != -1:
            print(f"Found the solution, ending...")
            self.solution_was_found = True
            return True

        if False not in solutions:
            print(f"All population is solution, ending...")
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
        n = self.dna_size

        cut = random.randint(0, n)
        genes1 = self.gene_block(parent1)
        genes2 = self.gene_block(parent2)

        genes1_left = genes1[:cut]
        genes2_left = genes2[:cut]

        genes1_right = genes1[cut:]
        genes2_right = genes2[cut:]

        genes1_ = genes1_left
        genes2_ = genes2_left

        for g in (genes2_right + genes2_left):
            if g not in genes1_left and len(genes1_) <= n:
                genes1_.append(g)

        for g in (genes1_right + genes1_left):
            if g not in genes2_left and len(genes2_) <= n:
                genes2_.append(g)

        # Dúvida?? Fizemos isso pois o tamanho retorna erradO!
        while len(genes1_) != n and len(genes1_) <= n:
            print(len(genes1_), n)
            i = random.choice(range(0,n))
            if i not in genes1_:
                genes1_.append(random.choice(genes2_right + genes2_left))

        while len(genes2_) != n and len(genes2_) <= n:
            i = random.choice(range(0,n))
            if i not in genes2_:
                genes2_.append(random.choice(genes1_right + genes1_left))

        # print(genes1_(genes2_)

        child1 = self.new_indiv_func(self.dna_size, dna=''.join(genes1_))

        child2 = self.new_indiv_func(self.dna_size, dna=''.join(genes2_))

        return child1, child2

    def single_mutation(self, indiv):
        genes = self.gene_block(indiv)
        index = random.randrange(0, len(genes))
        # print(genes)

        new_gene = [i for i in random.choices(self.gene_set, k=self.genotipe_size)]
        alternate = [i for i in random.choices(self.gene_set, k=self.genotipe_size)]
        while alternate == new_gene:
            alternate = [i for i in random.choices(self.gene_set, k=self.genotipe_size)]

        new_gene, alternate = ''.join(alternate), ''.join(new_gene) # Let random things happen!

        # print(alternate, new_gene)
        # print(index)

        genes[index] = alternate if new_gene == genes[index] else new_gene
        # print(genes)

        return ''.join(genes)

    def double_mutation(self, indiv):
        genes = self.gene_block(indiv)
        index = random.sample(range(len(genes)), 2)

        genes[index[0]], genes[index[1]] = genes[index[1]], genes[index[0]]

        return ''.join(genes)

    def pertube_mutation(self, indiv):
        genes = self.gene_block(indiv)
        index = sorted(random.sample(range(len(genes)), 2))

        # We rearrange the genes between the two indexes
        new_genes = genes[:index[0]] + random.sample(genes[index[0]:index[1]], index[1] - index[0]) + genes[index[1]:]

        return ''.join(new_genes)

    def insertion_mutation(self, indiv):
        genes = self.gene_block(indiv)
        index = sorted(random.sample(range(len(genes)), 2))
        if index[0] == index[1] + 1:
            index[1] += 1
        
        print(index[0], index[1])
        
        new_genes = genes[:index[0]] + genes[index[0]+1:index[1]] + [genes[index[0]]] + genes[index[1]:]
        print(new_genes)
        return ''.join(new_genes)

    def parent_tournament(self, population, choices_size, return_size):
        list_parents = random.sample(population, choices_size)
        best_parents = sorted(list_parents, key=lambda indiv: indiv.fitness, reverse=True)
        print(f"Selecting parents: {[i.fitness for i in best_parents]}", end='\n', sep=',')
        # Best parents:
        print(f"Chosen parents: {[i.fitness for i in best_parents][:return_size]}", end='\n', sep=',')

        return best_parents[:return_size]

    def parent_spinwheel(self, population, return_size):
        parents = []

        for _ in range(return_size):
            # fitness_sum = sum([indiv.fitness for indiv in population])
            fitnesses = [indiv.fitness for indiv in population]

            choice = random.choices(population, weights=fitnesses, k=1)[0]

            parents.append(choice)
            population.remove(choice)

        return parents

    def choose_survivor(self, return_size):
        bests = sorted(self.population, key=lambda indiv: indiv.fitness, reverse=True)

        for _ in range(len(self.population) - return_size):
            print("Removing worst: " + str(bests[-1].fitness))
            self.population.remove(bests[-1])
            bests.remove(bests[-1])


if __name__ == '__main__':
    g = Genetic(new_board, dna_size=16, population_size=5, genotipe_size=4)

    g.run()
    # for i in g.init_population()[:4]:
        # print(i.get_board())
        # print("Depois da mutação:")
        # i.dna = g.pertube_mutation(i)
        # print(i.get_board())
        # print()

    
    # g.run()
    # solutions = []

    # for indiv in g.population:
    #     if indiv.isSolution and indiv.dna not in [i.dna for i in solutions]:
    #         solutions.append(indiv)
    #         g.population.remove(indiv)

    # for s in solutions:
    #     s.show()
    # dna_1 = board_to_binary([0, 2, 4, 1, 5, 3, 6, 7], 8)
    # dna_2 = board_to_binary([7, 6, 5, 4, 3, 2, 1, 0], 8)
    # g.population[0] = new_board(8, dna_1) 
    # g.population[1] = new_board(8, dna_2)
    # parents = g.parent_tournament(g.population, 5, 2)
    # g.run()
    # g.crossover_cutandfill(g.population[0], g.population[1])
    # g = Genetic(new_board, mutation_method="single", parent_method="spinwheel", max_iterations=-1)
    # g.run()
    # g.run()
    # g.population = g.init_population()
    # g.choose_survivor(5)

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



