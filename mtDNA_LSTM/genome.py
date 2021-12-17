"""The genome to be evolved."""

import random
import numpy as np
import logging
import hashlib
import copy
import uuid
from train import train_lstm_model


class Genome():
    """
    Represents one genome and all relevant utility functions (add, mutate, etc.).
    """

    def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0,mtDNA = False):
  #  def __init__(self, all_possible_genes=None, geneparam={}, u_ID=0, mom_ID=0, dad_ID=0, gen=0):
        """Initialize a genome.

        Args:
            all_possible_genes (dict): Parameters for the genome, includes:
                gene_nb_neurons_i (list): [64, 128, 256]      for (i=1,...,6)
                gene_nb_layers (list):  [1, 2, 3, 4]
                gene_activation (list): ['relu', 'elu']
                gene_optimizer (list):  ['rmsprop', 'adam']
        """
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
        genh = str(self.nb_neurons())  \
               + str(self.geneparam['nb_layers']) + self.geneparam['optimizer']

        self.hash = hashlib.md5(genh.encode("UTF-8")).hexdigest()

        self.rmse = 0.0

    def set_genes_random(self):
        self.parents = [0, 0]  # very sad - no parents :(

        for key in self.all_possible_genes:
            self.geneparam[key] = random.choice(self.all_possible_genes[key])

        self.update_hash()


    def mutate_one_gene(self):
        gene_to_mutate = random.choice(list(self.all_possible_genes.keys()))

        current_value = self.geneparam[gene_to_mutate]
        possible_choices = copy.deepcopy(self.all_possible_genes[gene_to_mutate])

       # possible_choices.remove(current_value)

        self.geneparam[gene_to_mutate] = random.choice(possible_choices)

        self.update_hash()

    def set_generation(self, generation):
        """needed when a genome is passed on from one generation to the next.
        the id stays the same, but the generation is increased"""

        self.generation = generation
        # logging.info("Setting Generation to %d" % self.generation)

    def set_genes_to(self, geneparam, mom_ID, dad_ID):
        """Set genome properties.
        this is used when breeding kids

        Args:
            genome (dict): The genome parameters
        IMPROVE
        """
        self.parents = [mom_ID, dad_ID]

        self.geneparam = geneparam

        self.update_hash()

    def train(self):
        """Train the genome and record the accuracy.

        Args:
            dataset (str): Name of dataset to use.

        """
        if self.rmse == 0.0:  # don't bother retraining ones we already trained
      #      self.accuracy = train_and_score(self, trainingset)
            self.rmse = train_lstm_model(self)

    def print_genome(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%%" % (self.accuracy * 100))
        logging.info("UniID: %d" % self.u_ID)
        logging.info("Mom and Dad: %d %d" % (self.parents[0], self.parents[1]))
        logging.info("Gen: %d" % self.generation)
        # logging.info("Hash: %s" % self.hash)

    def print_genome_ma(self):
        """Print out a genome."""
        self.print_geneparam()
        logging.info("Acc: %.2f%% UniID: %d Mom and Dad: %d %d Gen: %d" % (
        self.accuracy * 100, self.u_ID, self.parents[0], self.parents[1], self.generation))
        logging.info("Hash: %s" % self.hash)

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
