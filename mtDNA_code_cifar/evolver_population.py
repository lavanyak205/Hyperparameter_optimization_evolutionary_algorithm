

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
    def __init__(self, all_possible_genes, elite_ratio=0.1, random_select=0.5, cross_over=0.9, mutate_chance=0.5):


        self.all_possible_genes = all_possible_genes
        self.elite_ratio = elite_ratio
        self.random_select = random_select
        self.cross_over = cross_over
        self.mutate_chance = mutate_chance

        self.ids = IDgen()

    def reset_mtDNA(self, pop, mtDNAstatus):
        logging.info("In Reset mtDNA")
        if mtDNAstatus:
            for genome in pop:
                genome.mtDNA = uuid.uuid4().hex
        return pop

    def create_population(self, count, mtDNAStatus):
        pop = []

        i = 0

        while i < count:

            # Initialize a new genome.
            genome = Genome(self.all_possible_genes, {}, self.ids.get_next_ID(), 0, 0, self.ids.get_Gen(), mtDNAStatus)

            # Set it to random parameters.
            genome.set_genes_random()

            if i == 0:
                # this is where we will store all genomes
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

        # self.master.print_all_genomes()

        # exit()

        return pop

    @staticmethod
    def fitness(genome):

        return genome.accuracy

    def grade(self, pop):
        summed = reduce(add, (self.fitness(genome) for genome in pop))
        return summed / float((len(pop)))

    def parent_selection(self, pop):
        # Tournament Selection
        parents = []
        for i in range(2):
            individuals = random.choices(pop, k=5)
            graded = [(self.fitness(genome), genome) for genome in individuals]
            parents = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        # Roulette
        # Rank based
        logging.info(parents[0].accuracy)
        logging.info(parents[1].accuracy)
        return parents[0], parents[1]

    def breed(self, mom, dad, mtDNA):

        children = []


        pcl = len(self.all_possible_genes)

        recomb_loc = random.randint(1, pcl - 1)

        # for _ in range(2): #make _two_ children - could also make more
        child1 = {}
        child2 = {}


        keys = list(self.all_possible_genes)
        keys = sorted(keys)  # paranoia - just to make sure we do not add unintentional randomization

        # *** CORE RECOMBINATION CODE ****
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

    def evolve(self, pop, mtDNA):

        # increase generation
        self.ids.increase_Gen()

        # Get scores for each genome
        graded = [(self.fitness(genome), genome) for genome in pop]

        # and use those scores to fill in the master list
        for genome in pop:
            self.master.set_accuracy(genome)

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]

        # Get the number we want to keep unchanged for the next cycle.
        elite_length = int(len(graded) * self.elite_ratio)
        logging.info("Number of Elite members %d" % elite_length)


        new_generation = graded[:elite_length]
        # cross over
        dataset = 'cifar-10'
        cross_over_length = int(self.cross_over * (len(graded) - elite_length))
        remaining_pop = len(graded) - elite_length - cross_over_length
        for i in range(0, cross_over_length, 2):
            parent1, parent2 = self.parent_selection(graded)
            babies = self.breed(parent1, parent2,mtDNA)
            for baby in babies:
                if self.random_select > random.random():
                    gtc = copy.deepcopy(baby)
                    gtc.mutate_one_gene()
                    gtc.train(dataset)
                    if gtc.accuracy > parent1.accuracy:
                        new_generation.append(gtc) if gtc not in new_generation else new_generation
                    else:
                        new_generation.append(parent1) if parent1 not in new_generation else new_generation
                    if gtc.accuracy > parent2.accuracy:
                        new_generation.append(gtc) if gtc not in new_generation else new_generation
                    else:
                        new_generation.append(parent2) if parent2 not in new_generation else new_generation

                else:
                    baby.train(dataset)
                    if baby.accuracy > parent1.accuracy:
                        new_generation.append(baby) if baby not in new_generation else new_generation
                    else:
                        new_generation.append(parent1) if parent1 not in new_generation else new_generation
                    if baby.accuracy > parent2.accuracy:
                        new_generation.append(baby) if baby not in new_generation else new_generation
                    else:
                        new_generation.append(parent2) if parent2 not in new_generation else new_generation

        # Mutate left over population members
        for genome in graded[-remaining_pop:]:
            if self.random_select > random.random():
                gtc = copy.deepcopy(genome)

                while self.master.is_duplicate(gtc):
                    gtc.mutate_one_gene()

                gtc.set_generation(self.ids.get_Gen())
                new_generation.append(gtc)
                self.master.add_genome(gtc)

        return new_generation, graded[0].accuracy * 100, graded[-1].accuracy * 100
