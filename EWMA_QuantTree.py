from extendedQuantTree import Extended_Quant_Tree
from neuralNetworks import NN_man
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import extendedQuantTree as ext
import qtLibrary.libquanttree as qt



class EWMA_QuantTree:

    def __init__(self, initial_pi_values, lamb, statistic, maxN, minN, alpha, stop, does_restart, nu, beta):
        self.tree = Extended_Quant_Tree(ext.create_bins_combination(len(initial_pi_values), 2 * len(initial_pi_values)))
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
        self.record_history = []

    def initialize(self, training_set):
        if self.statistic ==  qt.pearson_statistic:
            p = 0
        elif self.statistic == qt.tv_statistic:
            p = 1
        else:
            raise ('Strange exception: ' + str(self.statistic))
        self.tree.build_histogram(training_set)
        self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).estimate_quanttree_threshold(self.alpha, 10000)

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
        return values[index]

    def classic_batch_analysis(self, batch):
        stat = self.statistic(self.tree, batch)
        positive =  stat > self.threshold
        self.record_history.append(positive)
        return positive

    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        positive = self.classic_batch_analysis(batch)
        self.value = (1 - self.lamb) * self.value + positive * self.lamb
        return self.value

    def find_change(self, EMWA_value, batch):
        # If there is a change we return True
        return EMWA_value > self.threshold

