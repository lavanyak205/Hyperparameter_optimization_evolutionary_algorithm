
import math

from evolver_population import Evolver

from tqdm import tqdm

import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='LSTM.txt'
)


def train_members(genomes):
    pbar = tqdm(total=len(genomes))
    for genome in genomes:
        genome.train()
        pbar.update(1)

    pbar.close()

def generation_best(genomes):
    graded = [(genome.rmse, genome) for genome in genomes]
    graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=False)]
    return graded[0].rmse, graded[-1].rmse


def get_average_rmse(genomes):
    total_rmse = 0

    for genome in genomes:
        total_rmse += genome.rmse

    return total_rmse / len(genomes)

def generate_ea(generations, population, all_possible_genes, mtDNA=False):

    evolver = Evolver(all_possible_genes)

    genomes = evolver.create_population(population, mtDNA)

    rmse_gen = {}
    best_pop_rmse = {}
    worst_pop_rmse = {}
    # Evolve the generation.
    for i in range(generations):


        logging.info("***Now in generation %d of %d***" % (i + 1, generations))
        train_members(genomes)

        average_rmse = get_average_rmse(genomes)
        rmse_gen[i + 1] = average_rmse
        # Print out the average accuracy each generation.
        logging.info("Generation RMSE: %.2f%%" % (average_rmse) )
        logging.info('-' * 80)

        # Evolve, except on the last iteration.
        if i != generations - 1:

            genomes, best_pop_rmse[i+1], worst_pop_rmse[i+1] = evolver.evolve(genomes, mtDNA)
        else:
            best_pop_rmse[i + 1], worst_pop_rmse[i + 1] = generation_best(genomes)

        if i+1 == int(math.log2(population)) and mtDNA:
            genomes = evolver.reset_mtDNA(genomes, mtDNA)
    genomes = sorted(genomes, key=lambda x: x.rmse, reverse=False)
    logging.info(genomes[0].geneparam)
    logging.info("Best member in generation: RMSE: %.2f%%" % (genomes[0].rmse))
    logging.info(genomes[-1].geneparam)
    logging.info("Worst member in generation: RMSE: %.2f%%" % (genomes[-1].rmse))

    return rmse_gen, best_pop_rmse, worst_pop_rmse


def main():
    """Evolve a genome."""
    population = 30  # Number of networks/genomes in each generation.
    # we only need to train the new ones....

    generations = 8 # Number of times to evolve the population.
    possible_genes = {
        'nb_neurons': [16, 32, 64, 128, 256],
        'nb_layers': [1, 2],
        'nb_batch_size': [32, 64],
        'n_window_size':[144, 432, 720],
        'n_epoch':  [10, 15],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad']
    }

    # replace nb_neurons with 1 unique value for each layer
    # 6th value reserved for dense layer
    nb_neurons = possible_genes['nb_neurons']
    for i in range(1, 6):
        possible_genes['nb_neurons_' + str(i)] = nb_neurons

    possible_genes.pop('nb_neurons')

    print("***Evolving for %d generations with population size = %d***" % (generations, population))

    setmtDNA = True

    generation_rmse = generate_ea(generations, population, possible_genes, setmtDNA)

    logging.info("Generation accuracy")
    logging.info(generation_rmse)

if __name__ == '__main__':
    main()
