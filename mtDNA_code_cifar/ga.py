"""Entry point to evolving the neural network. Start here."""

from evolver_population import Evolver

from tqdm import tqdm
import time
import logging
import math
import numpy as np
import matplotlib.pyplot as plt

import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO,
    filename='cifar_Dec13_Tournament.txt'
)


def train_genomes(genomes, dataset):
    logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(dataset)
        pbar.update(1)

    pbar.close()



def generation_best(genomes):
    graded = [(genome.accuracy, genome) for genome in genomes]
    graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
    return graded[0].accuracy * 100, graded[-1].accuracy * 100


def get_genome_individual(genomes, index):
    return genomes[index]


def get_average_accuracy(genomes):
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes)


def generate_ea(generations, population, possible_genes, dataset, mtDNA=False):
    logging.info("***generate(generations, population, all_possible_genes, dataset)***")

    evolve = Evolver(possible_genes)

    genomes = evolve.create_population(population, mtDNA)
    accuracy_gen = {}
    best_acc = {}
    worst_acc = {}
    # Evolve the generation.
    for i in range(generations):

        logging.info("***generation %d of %d***" % (i + 1, generations))

        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)
        average_accuracy = get_average_accuracy(genomes)
        accuracy_gen[i + 1] = average_accuracy
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-' * 80)  # -----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes, best_acc[i + 1], worst_acc[i + 1] = evolve.evolve(genomes,mtDNA)
        else:
            best_acc[i + 1], worst_acc[i + 1] = generation_best(genomes)
        if mtDNA:
            if i + 1 == int(math.log2(population)):
                genomes = evolve.reset_mtDNA(genomes, mtDNA)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)
    logging.info("Best member in population accuracy: %.2f%%" % (genomes[0].accuracy * 100))
    logging.info(genomes[0].geneparam)
    logging.info("Worst member in population accuracy: %.2f%%" % (genomes[-1].accuracy * 100))
    logging.info(genomes[-1].geneparam)
    # Print out the top 5 networks/genomes.

    # save_path = saver.save(sess, '/output/model.ckpt')
    # print("Model saved in file: %s" % save_path)
    return accuracy_gen, best_acc, worst_acc


def main():
    population = 30  # Number of initial population

    dataset = 'cifar10_cnn'

    generations = 8  # Number of times to evolve the population.

    possible_genes = {
        'nb_neurons': [16, 32, 64, 128, 256],
        'nb_layers': [1, 3, 5],
        'nb_batch_size': [64, 128],
        'n_epoch': [128, 256],
        'activation': ['relu', 'elu', 'tanh', 'softplus'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad']
    }

    # replace nb_neurons with 1 unique value for each layer
    # 6th value reserved for dense layer
    nb_neurons = possible_genes['nb_neurons']
    for i in range(1, 6):
        possible_genes['nb_neurons_' + str(i)] = nb_neurons
    # remove old value from dict
    possible_genes.pop('nb_neurons')
    mtDNA = True
    print("***Evolving for %d generations with population size = %d***" % (generations, population))
    generation_acuracy, best_acc, worst_acc = generate_ea(generations, population, possible_genes, dataset, mtDNA)
    # For debugging
    logging.info("Generation accuracy")
    logging.info(generation_acuracy)
    logging.info("Best Population accuracy")
    logging.info(best_acc)
    logging.info('worst accuracy')
    logging.info(worst_acc)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    logging.info("Execution Time: %s" % str(end - start))
