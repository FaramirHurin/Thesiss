from extendedQuantTree import Extended_Quant_Tree
from neuralNetworks import NN_man
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import extendedQuantTree as ext
import qtLibrary.libquanttree as qt



class EWMA_QuantTree:

    def __init__(self, initial_pi_values, lamb, statistic, maxN, minN, alpha, stop, does_restart, nu, beta):
        pi_values = ext.create_bins_combination(len(initial_pi_values), len(initial_pi_values))
        self.pi_values = pi_values
        self.tree = Extended_Quant_Tree(pi_values)
        self.lamb = lamb
        self.nu = nu
        self.statistic = statistic
        self.record_history = []
        self.alpha = [0.4]
        self.value = self.alpha[0]
        self.beta = 0.01  # 1/ARL0
        self.threshold = 0
        self.max_len_computed = 0
        self.EWMA_threshold = 0
        self.EWMA_thresholds = 0
        self.stop = stop

    def initialize(self, training_set):
        self.trainig_set = training_set
        if self.statistic ==  qt.pearson_statistic:
            p = 0
        elif self.statistic == qt.tv_statistic:
            p = 1
        else:
            raise ('Strange exception: ' + str(self.statistic))
        self.tree.build_histogram(training_set)
        self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
            estimate_quanttree_threshold(self.alpha, 10000)
        """
        val = 0
        exp = 1
        for index in range(exp):
            val += self.compute_EWMA_threshold()
        self.EWMA_threshold = val/exp
        print(self.EWMA_threshold)
        """
    # MC simulation, constant threshold
    def compute_EWMA_threshold(self):
        run_lenght = 1000000
        tree, threshold, ewma_0 = self.prepare_simulated_run()
        values = self.simulate_EWMA_run(run_lenght, tree, ewma_0)
        selection = values[:int(run_lenght*self.beta)]
        return np.max(selection)

    def alterative_EWMA_thresholds_computation(self):
        x = None
        experiments = 2000
        max_lenght = 20
        table = self.fill_table(experiments, max_lenght)
        means = np.zeros(table.shape[1])
        for index in range(len(means)):
            means[index] = np.mean(table[:,index])
        print('means are' + str(means))
        self.EWMA_thresholds = self.compute_thresholds_on_table(table, None)
        self.max_len_computed = len(self.EWMA_thresholds)
        return self.EWMA_thresholds

    def compute_thresholds_on_table(self, table, thresholds):
        if table.shape[0] < 1/self.beta or table.shape[1] < 1:
            return thresholds
        if thresholds is None:
            thresholds = []
        values = table[:, 0]
        vals = np.sort(values)
        threshold = vals[int(len(vals) * (1 - self.beta))]
        if threshold < np.mean(vals) - 0.0001: #Costante necessaria per gli arrotondamenti
            print('Vals are: ' + str(vals))
            print('Threshold is' + str(threshold))
            raise('Bad threshold comutation')
        thresholds.append(threshold)
        to_eliminate = []
        for index in range(table.shape[0]):
            if table[index, 0] > threshold:
                to_eliminate.append(index)
        table = np.delete(table, to_eliminate, axis=0)
        table = np.delete(table, 0, axis = 1)
        thresholds = self.compute_thresholds_on_table(table, thresholds)
        return thresholds

    def fill_table(self, experiments, max_lenght):
        indipendence_percentage = 0.0001
        table = np.zeros([experiments, max_lenght - 1])
        for index in range(experiments):
            if index % 10 == 0:
                print(index)
                if index * indipendence_percentage %1 == 0:
                    tree, ewma_0 = self.prepare_simulated_run()
                    print('simulated')
            values = self.simulate_EWMA_run(max_lenght, tree, ewma_0)
            table[index] = values[1:]
        table[:, 0] = ewma_0
        return table

    def prepare_simulated_run(self):
        training_set = np.random.uniform(0, 1, self.tree.ndata)
        tree = qt.QuantTreeUnivariate(self.tree.pi_values)
        tree.build_histogram(training_set)
        poses = 0
        for index in range(4000):
            batch = np.random.uniform(0, 1, self.nu)
            positive = self.classic_batch_analysis(batch, tree)
            poses += positive
        ewma_0 = poses/index
        return tree, ewma_0

    def simulate_EWMA_run(self, max_lenght, tree, ewma_0):
        values = np.zeros(max_lenght)
        values[0] = ewma_0
        for index in range(1, max_lenght):
            batch = np.random.uniform(0, 1, self.nu)
            positive = self.classic_batch_analysis(batch, tree)
            values[index] = (1 - self.lamb) * values[index - 1] + positive * self.lamb
        return values

    def classic_batch_analysis(self, batch, tree):
        if tree is None:
            tree = self.tree
        stat = self.statistic(tree, batch)
        positive =  stat > self.threshold
        self.record_history.append(positive)
        return positive
    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        positive = self.classic_batch_analysis(batch)
        self.value = (1 - self.lamb) * self.value + positive * self.lamb
        return self.value

    def find_change(self):
        # If there is a change we return True
        return self.value > self.EWMA_threshold
