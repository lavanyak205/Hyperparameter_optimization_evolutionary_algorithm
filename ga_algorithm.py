"""Entry point to evolving the neural network. Start here."""

from evolver import Evolver

from tqdm import tqdm

import logging
import numpy as np
import matplotlib.pyplot as plt

import sys

# Setup logging.
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
    level=logging.INFO  # ,
    # filename='log.txt'
)


def train_genomes(genomes, dataset):
    """Train each genome.

    Args:
        networks (list): Current population of genomes
        dataset (str): Dataset to use for training/evaluating

    """
   # logging.info("***train_networks(networks, dataset)***")

    pbar = tqdm(total=len(genomes))

    for genome in genomes:
        genome.train(dataset)
        pbar.update(1)

    pbar.close()

def get_genome_individual(genomes, index):
    return genomes[index]


def get_average_accuracy(genomes):
    """Get the average accuracy for a group of networks/genomes.

    Args:
        networks (list): List of networks/genomes

    Returns:
        float: The average accuracy of a population of networks/genomes.

    """
    total_accuracy = 0

    for genome in genomes:
        total_accuracy += genome.accuracy

    return total_accuracy / len(genomes)


#def generate(generations, population, all_possible_genes, dataset, mtDNA = False):
def generate(generations, population, all_possible_genes, dataset, mtDNA=False):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evolve the population
        population (int): Number of networks in each generation
        all_possible_genes (dict): Parameter choices for networks
        dataset (str): Dataset to use for training/evaluating

    """
   # logging.info("***generate(generations, population, all_possible_genes, dataset)***")

    evolver = Evolver(all_possible_genes)

    genomes = evolver.create_population(population, mtDNA)
    accuracy_gen = {}
    # Evolve the generation.
    for i in range(generations):

        logging.info("***Now in generation %d of %d***" % (i + 1, generations))

        #print_genomes(genomes)

        # Train and get accuracy for networks/genomes.
        train_genomes(genomes, dataset)
        if mtDNA:
        # Pick 2 parents based on fitness values
            first_parent = get_genome_individual(genomes, 1)
        # Generate a random number to pick second parent
            pick_parent = np.random.randint(1, population+1)
            second_parent = get_genome_individual(genomes, pick_parent)
        #Check if parents have same mtDNA
            if(first_parent.mtDNA == second_parent.mtDNA):
            # Don't allow for cross-over
                pass
            else:
                pass
                # Allow for cross-over
        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(genomes)
        accuracy_gen[i + 1] = average_accuracy
        # Print out the average accuracy each generation.
        logging.info("Generation average: %.2f%%" % (average_accuracy * 100))
        logging.info('-' * 80)  # -----------

        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Evolve!
            genomes = evolver.evolve(genomes)

    # Sort our final population according to performance.
    genomes = sorted(genomes, key=lambda x: x.accuracy, reverse=True)

    # Print out the top 5 networks/genomes.
    print_genomes(genomes[:5])

    # save_path = saver.save(sess, '/output/model.ckpt')
    # print("Model saved in file: %s" % save_path)
    return accuracy_gen

def print_genomes(genomes):
    """Print a list of genomes.

    Args:
        genomes (list): The population of networks/genomes

    """
  #  logging.info('-' * 80)

    for genome in genomes:
        genome.print_genome()


def main():
    """Evolve a genome."""
    population = 2 # Number of networks/genomes in each generation.
    # we only need to train the new ones....


    dataset = 'cifar10_cnn'


    print("***Dataset:", dataset)
    generations = 1  # Number of times to evolve the population.
    all_possible_genes = {
        'nb_neurons': [16, 32, 64, 128],
        'nb_layers': [1, 2, 3, 4, 5],
        'nb_batch_size': [32, 64, 128],
        'n_epoch': [128, 256],
        'activation': ['relu', 'elu', 'tanh', 'softplus'],
        'optimizer': ['rmsprop', 'adam', 'sgd', 'adagrad',]
    }


    # replace nb_neurons with 1 unique value for each layer
    # 6th value reserved for dense layer
    nb_neurons = all_possible_genes['nb_neurons']
    for i in range(1, 5):
        all_possible_genes['nb_neurons_' + str(i)] = nb_neurons
    # remove old value from dict
    all_possible_genes.pop('nb_neurons')

    print("***Evolving for %d generations with population size = %d***" % (generations, population))
    # if mtDNA is enabled
    # setmtDNA = False
    # if setmtDNA:
    #     generation_acuracy = generate(generations, population, all_possible_genes, dataset,setmtDNA)
    # else:
    generation_acuracy = generate(generations, population, all_possible_genes, dataset)
    print('generation accuracy')
    print(generation_acuracy)
    #Plot the generations vs accuracy
    plt.plot(generation_acuracy.keys(), generation_acuracy.values(), 'r*')
    plt.show()
if __name__ == '__main__':
    main()
