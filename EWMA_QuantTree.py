from extendedQuantTree import Extended_Quant_Tree
from neuralNetworks import NN_man
import numpy as np
import pandas as pd
import logging, sys
import matplotlib.pyplot as plt
import qtLibrary.libquanttree as qt

logging.basicConfig(stream=sys.stderr, level=logging.ERROR)


class EWMA_QuantTree:

    def __init__(self, initial_pi_values, lamb, statistic, maxN, minN, alpha, stop, does_restart, nu, beta):
        self.tree = qt.QuantTree(initial_pi_values)
        self.initial_pi_values = initial_pi_values
        self.lamb = lamb
        self.nu = nu
        self.statistic = statistic
        self.alpha = alpha
        self.value = self.alpha[0]
        self.beta = beta  # sort of alpha for the second level statistic
        self.status = 0
        # self.threshold = self.compute_EWMA_threshold(nu)
        self.threshold = 0
        self.stop = stop
        self.does_restart = does_restart
        self.buffer = None

    def initialize(self, training_set):
        self.tree.build_histogram(training_set)
        self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).estimate_quanttree_threshold(self.alpha, 6000)

    # MC simulation
    def compute_EWMA_threshold(self, nu):
        lenght = 50000
        statistics = np.zeros(lenght)
        # Uses its own methods to simulate a run and compute the threshold
        self.tree.build_histogram(np.random.random(nu))
        for round in range(lenght):
            batch = np.random.random(nu)
            statistics[round] = self.compute_EMWA(batch)
        values = np.sort(statistics)
        index = int((1 - self.beta) * lenght)
        # Restores original values
        self.tree = qt.QuantTree(self.initial_pi_values)  # Come back to original Pi values
        logging.debug('EMWA threshold is ' + str(values[index]))
        return values[index]

    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        stat = self.statistic(self.tree, batch)
        positive = stat > self.threshold
        self.value = (1 - self.lamb) * self.value + positive * self.lamb
        # logging.debug('value is' + str(self.value))
        return self.value

    def find_change(self, EMWA_value, batch):
        # If there is a change we return True
        if EMWA_value > self.threshold:
            logging.debug('change found')
            if self.does_restart:
                self.restart()
            return True
        else:
            return False


    def compute_EMWA_threshold_with_a_stat(self, statistic, nu):
        lenght = 10000
        statistics = np.zeros(lenght)
        # Uses its own methods to simulate a run and compute the threshold
        if statistic == 'uniform':
            batch = np.random.uniform(0, 1, nu)
        elif statistic == 'normal':
            batch = np.random.normal(0, 1, nu)
        self.tree.build_histogram(batch)
        for round in range(lenght):
            if statistic == 'uniform':
                batch = np.random.uniform(0, 1, nu)
            elif statistic == 'normal':
                batch = np.random.normal(0, 1, nu)
            statistics[round] = self.compute_EMWA(batch)
        #values = np.sort(statistics)
        index = int((1 - self.beta) * lenght)

        # Restores original values
        self.tree = Extended_Quant_Tree(self.initial_pi_values)  # Come back to original Pi values
        self.value = self.alpha  # Reset stat value to the original value
        #logging.debug('EMWA threshold is ' + str(values[index]))

        # serie = pd.Series(statistics, name='Stat value when threshold <alpha')
        # serie.plot()

        # plt.legend('len is' + str(len(serie)))
        # plt.show()
        logging.debug(np.mean(statistics))
        return statistics


class Static_EMWA_QuantTree():
    def __init__(self, initial_pi_values, lamb, statistic, maxN, minN, alpha, stop, nu, beta):
        self.tree = Extended_Quant_Tree(initial_pi_values)
        self.initial_pi_values = initial_pi_values
        self.lamb = lamb
        self.statistic = statistic
        self.alpha = alpha[0]
        self.value = self.alpha
        self.nu = nu
        self.beta = beta  # sort of alpha for the second level statistic
        self.NN = NN_man(len(initial_pi_values), maxN, minN, 300)
        # self.NN.train(alpha)
        self.status = 0
        # self.threshold = self.compute_EWMA_threshold(nu)
        tree = qt.QuantTree(initial_pi_values)
        tree.ndata = 4000
        #self.threshold = qt.ChangeDetectionTest(tree, self.nu, self.statistic).estimate_quanttree_threshold(alpha, 10000)
        self.threshold = 0
        self.stop = stop

    # MC simulation
    def compute_EWMA_threshold(self, nu):
        lenght = 5000
        statistics = np.zeros(lenght)
        # Uses its own methods to simulate a run and compute the threshold
        self.tree.build_histogram(np.random.random(nu))
        for round in range(lenght):
            batch = np.random.random(nu)
            statistics[round] = self.compute_EMWA(batch)
        values = np.sort(statistics)
        index = int((1 - self.beta) * lenght)
        # Restores original values
        self.tree = Extended_Quant_Tree(self.initial_pi_values)  # Come back to original Pi values
        self.value = self.alpha  # Reset stat value to the original value
        logging.debug('EMWA threshold is ' + str(values[index]))
        return values[index]

    def create_tree(self, training_set):
        self.tree.build_histogram(training_set)
        return

    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        stat = self.statistic(self.tree, batch)
        # threshold = self.NN.predict_value(self.tree.pi_values, self.tree.ndata)
        positive = stat > self.threshold
        self.value = (1 - self.lamb) * self.value + positive * self.lamb
        # logging.debug('value is' + str(self.value))
        return self.value

    def normal_acceptance(self, batch):
        stat = self.statistic(self.tree, batch)
        # threshold = self.NN.predict_value(self.tree.pi_values, self.tree.ndata)
        positive = stat > self.threshold
        return positive

    def find_change(self, EMWA_value, batch):
        # If there is a change we return True
        if EMWA_value > self.threshold:
            logging.debug('change found')
            return True
        else:
            return False

    def playRound(self, batch):
        EMWA = self.compute_EMWA(batch)
        # If we have to stop when we find a change and we think there is one, we exit
        if self.stop and self.find_change(EMWA, batch):
            return EMWA, True
        self.status += 1
        return EMWA, False

    def compute_EMWA_threshold_with_a_stat(self, statistic, N):
        lenght = 4000
        statistics = np.zeros(lenght)
        # Uses its own methods to simulate a run and compute the threshold
        if statistic == 'uniform':
            training_set = np.random.uniform(0, 1, N)
        elif statistic == 'normal':
            training_set = np.random.normal(0, 1, N)
        self.tree.build_histogram(training_set)
        self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
            estimate_quanttree_threshold([self.alpha], 4000)
        for round in range(lenght):
            if statistic == 'uniform':
                batch = np.random.uniform(0, 1, self.nu)
            elif statistic == 'normal':
                batch = np.random.normal(0, 1, self.nu)
            statistics[round] = self.compute_EMWA(batch)
        values = np.sort(statistics)
        index = int((1 - self.beta) * lenght)
        # Restores original values
        self.tree = Extended_Quant_Tree(self.initial_pi_values)  # Come back to original Pi values
        self.value = self.alpha  # Reset stat value to the original value
        logging.debug('EMWA threshold is ' + str(values[index]))
        serie = pd.Series(statistics, name='Stat value when threshold <alpha')
        # serie.plot()
        # plt.legend('len is' + str(len(serie)))
        # plt.title('True static EMWA')
        # plt.show()
        logging.debug(np.mean(values))
        return statistics

    def tell_averaggio(self, statistic, N):
        lenght = 4000
        statistics = np.zeros(lenght)
        if statistic == 'uniform':
            training_set = np.random.uniform(0, 1, lenght)
        elif statistic == 'normal':
            training_set = np.random.normal(0, 1, lenght)
        self.tree.build_histogram(training_set)
        self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
            estimate_quanttree_threshold(self.alpha, 4000)
        for round in range(lenght):
            if statistic == 'uniform':
                batch = np.random.uniform(0, 1, self.nu)
            elif statistic == 'normal':
                batch = np.random.normal(0, 1, self.nu)
            statistics[round] = self.normal_acceptance(batch)
        return (np.average(statistics))
