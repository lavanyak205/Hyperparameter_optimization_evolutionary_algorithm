
import random
import logging
import copy
from functools import reduce
from operator import add
from genome import Genome
from id_generator import IDgen
from allgenomes import AllGenomes
import uuid


class Evolver():
    def __init__(self, all_possible_genes, elite_ratio=0.1, random_select=0.5, cross_over_chance = 0.9, mutate_chance=0.75):

        self.all_possible_genes = all_possible_genes
        self.elite_ratio = elite_ratio
        self.random_select = random_select
        self.cross_over_chance = cross_over_chance
        self.mutate_chance = mutate_chance

        self.ids = IDgen()

    def create_population(self, count, mtDNAStatus):
        pop = []

        i = 0

        while i < count:

            # Initialize a new genome.
            genome = Genome(self.all_possible_genes, {}, self.ids.get_next_ID(), 0, 0, self.ids.get_Gen(), mtDNAStatus)
            genome.set_genes_random()

            if i == 0:
                self.master = AllGenomes(genome)
            else:
                # Make sure it is unique....
                while self.master.is_duplicate(genome):
                    genome.mutate_one_gene()

            # Add the genome to our population.
            pop.append(genome)

            # and add to the master list
            if i > 0:
                self.master.add_genome(genome)

            i += 1



        return pop

    @staticmethod
    def fitness(genome):
        """Return the rmse, which is our fitness function."""
        return genome.rmse


    def parent_selection(self, pop):
        # Tournament Selection
        # Pick two parents
        parents = []
        for i in range(2):
            individuals = random.sample(pop, k=2)
            graded = [(self.fitness(genome), genome) for genome in individuals]
            individuals = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
            parents.append(individuals[0])
        # Roulette

        # for i in range(2):
        #     sum = 0
        #     sum_roulette = 0
        #     random.shuffle(pop)
        #     for genome in pop:
        #         sum = sum + genome.accuracy
        #     selection_point = random.uniform(0,sum)
        #     sum_roulette =0
        #     index = 0
        #     while(sum_roulette < selection_point):
        #         sum_roulette = sum_roulette + pop[index].accuracy
        #         index = index + 1
        #     if index == len(pop):
        #         index = len(pop)-1
        #     parents.append(pop[index])
        # #Rank based
        return parents[0], parents[1]

    def breed(self, dad, mom, mtDNA = False):

        children = []

        pcl = len(self.all_possible_genes)

        recomb_loc = random.randint(1, pcl - 1)

        # for _ in range(2): #make _two_ children - could also make more
        child1 = {}
        child2 = {}

        keys = list(self.all_possible_genes)
        keys = sorted(keys)  # paranoia - just to make sure we do not add unintentional randomization

        for x in range(0, pcl):
            if x < recomb_loc:
                child1[keys[x]] = mom.geneparam[keys[x]]
                child2[keys[x]] = dad.geneparam[keys[x]]
            else:
                child1[keys[x]] = dad.geneparam[keys[x]]
                child2[keys[x]] = mom.geneparam[keys[x]]


        genome1 = Genome(self.all_possible_genes, child1, self.ids.get_next_ID(), mom.u_ID, dad.u_ID,
                         self.ids.get_Gen())

        genome2 = Genome(self.all_possible_genes, child2, self.ids.get_next_ID(), mom.u_ID, dad.u_ID,
                         self.ids.get_Gen())
        if mtDNA:
            genome1.mtDNA = mom.mtDNA
            genome2.mtDNA = mom.mtDNA


        # at this point, there is zero guarantee that the genome is actually unique

        # Randomly mutate one gene
        if self.mutate_chance > random.random():
            genome1.mutate_one_gene()

        if self.mutate_chance > random.random():
            genome2.mutate_one_gene()

        # do we have a unique child or are we just retraining one we already have anyway?
        while self.master.is_duplicate(genome1):
            genome1.mutate_one_gene()

        self.master.add_genome(genome1)

        while self.master.is_duplicate(genome2):
            genome2.mutate_one_gene()

        self.master.add_genome(genome2)

        children.append(genome1)
        children.append(genome2)

        return children
    def reset_mtDNA(self, pop, mtDNAstatus):
        if mtDNAstatus:
            for genome in pop:
                genome.mtDNA = uuid.uuid4().hex
        return pop
    def evolve(self, pop, mtDNAstatus=False):

        # increase generation
        self.ids.increase_Gen()

        # Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in pop]

        # and use those scores to fill in the master list
        for genome in pop:
            self.master.set_rmse(genome)
        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
        graded = graded[:30]
        logging.info("Best population member fitness %s" % str(graded[0].rmse) )
        logging.info("Worst population member fitness %s" % str(graded[-1].rmse))


        # Get the number we want to keep unchanged for the next cycle.
        elite_length = int(len(graded) * self.elite_ratio)
        logging.info("Elite population length %d" % elite_length)

        new_generation = graded[:elite_length]
        cross_over_ratio = int(self.cross_over_chance * (len(graded) - elite_length))
        remain_members = len(graded) - elite_length - cross_over_ratio


        for i in range(0, cross_over_ratio, 2):
            parent1, parent2 = self.parent_selection(graded)
            babies = self.breed(parent1, parent2 ,mtDNAstatus)

            for baby in babies:
                if self.random_select > random.random():
                    gtc = copy.deepcopy(baby)
                    gtc.mutate_one_gene()
                    gtc.train()
                    if gtc.rmse < parent1.rmse:
                        new_generation.append(gtc)
                    else:
                        new_generation.append(parent1) if parent1 not in new_generation else new_generation
                    if gtc.rmse < parent2.rmse:
                        new_generation.append(gtc) if gtc not in new_generation else new_generation
                    else:
                        new_generation.append(parent2) if parent2 not in new_generation else new_generation

                else:
                    baby.train()
                    if baby.rmse < parent1.rmse:
                        new_generation.append(baby)
                    else:
                        new_generation.append(parent1) if parent1 not in new_generation else new_generation
                    if baby.rmse < parent2.rmse:
                        new_generation.append(baby) if baby not in new_generation else new_generation
                    else:
                        new_generation.append(parent2) if parent2 not in new_generation else new_generation

        # Mutate the left over population members
        for genome in graded[-remain_members:]:
            if self.random_select > random.random():
                gtc = copy.deepcopy(genome)

                while self.master.is_duplicate(gtc):
                    gtc.mutate_one_gene()

                gtc.set_generation(self.ids.get_Gen())
                new_generation.append(gtc)

        return new_generation, graded[0].rmse, graded[-1].rmse
