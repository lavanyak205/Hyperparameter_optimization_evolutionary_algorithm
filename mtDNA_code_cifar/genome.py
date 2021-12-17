

import random
import logging
import hashlib
import copy
import uuid
from train_ga import train_and_score


class Genome():
    def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0,mtDNA = False):

        self.accuracy = 0.0
        self.all_possible_genes = all_possible_genes
        self.geneparam = geneparam  # (dict): represents actual genome parameters

        self.parents = [mom_ID, dad_ID]
        self.generation = gen
        self.u_ID = u_ID
        if mtDNA:
            self.mtDNA = uuid.uuid4().hex
        # hash only makes sense when we have specified the genes
        if not geneparam:
            self.hash = 0
        else:
            self.update_hash()

    def update_hash(self):

        genh = str(self.nb_neurons()) + self.geneparam['activation'] \
               + str(self.geneparam['nb_layers']) + self.geneparam['optimizer']

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

        self.accuracy = 0.0

    def set_genes_random(self):

        # print("set_genes_random")
        self.parents = [0, 0]  # very sad - no parents :(

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])

        self.update_hash()

    def mutate_one_gene(self):


        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        # And then let's mutate one of the genes.
        # Make sure that this actually creates mutation
        current_value = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])

        possible_choices.remove(current_value)

        self.geneparam[gene_to_mutate] = random.choice(possible_choices)

        self.update_hash()

    def set_generation(self, generation):

        self.generation = generation


    def set_genes_to(self, geneparam, mom_ID, dad_ID):

        self.parents = [mom_ID, dad_ID]

        self.geneparam = geneparam

        self.update_hash()

    def train(self, trainingset):

        if self.accuracy == 0.0:  # don't bother retraining ones we already trained
            self.accuracy = train_and_score(self, trainingset)

    def print_genome(self):

        self.print_geneparam()


    def print_genome_ma(self):

        self.print_geneparam()


    # print nb_neurons as single list
    def print_geneparam(self):
        g = self.geneparam.copy()
        nb_neurons = self.nb_neurons()
        if not self.parents == [0,0]:
            for i in range(1, 6):
                g.pop('nb_neurons_' + str(i))
        # replace individual layer numbers with single list
            g['nb_neurons'] = nb_neurons
        # logging.info(g)

    # convert nb_neurons_i at each layer to a single list
    def nb_neurons(self):
        nb_neurons = [None] * 5

        if not self.all_possible_genes:
            nb_neurons = self.geneparam['nb_neurons']
        else:
            for i in range(0, 5):
                nb_neurons[i] = self.geneparam['nb_neurons_' + str(i + 1)]

        return nb_neurons
