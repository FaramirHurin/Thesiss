import qtLibrary.libquanttree as qt
import extendedQuantTree as aux
import numpy as np
from copy import copy
import logging, sys
import neuralNetworks

bins_number = 4
initial_pi_values = np.ones(bins_number)/bins_number
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)


class Quant_garden:

    def __init__(self, K, statistic, nu, alpha, beta, bins_number, min_N, max_N, uses_double_check, inner_alpha):
        nodes = 200
        self.K = K
        self.garden =  np.empty(K, dtype = aux.Incremental_Quant_Tree) #The trees
        self.batch_on_trial = None
        self.table = np.zeros([K, K])
        self.statistic = statistic
        self.nu = nu
        self.alpha = alpha # FP0 for the last tree
        self.inner_alpha = inner_alpha
        self.uses_double_check = uses_double_check
        # We might pickle an already trained NN, at least for tests
        self.last_neural = neuralNetworks.NN_man(bins_number, max_N, min_N, nodes) #for standard test
        self.last_neural.train(self.alpha)  #Verificare che la NN usi l'alhpa giusto
        self.hidden_neural = neuralNetworks.NN_man(bins_number, max_N, min_N, nodes) #for the hidden tests, with alpha2
        self.hidden_neural.train(0.5)
        self.status = 0 #Wether the algorithm is already fully running
        self.beta = beta # FP0 for hidden testing
        if uses_double_check:
            self.binomial_treshold = self.compute_threshold_for_binomial_test()
            self.hidden_thresholds = np.zeros(self.K)
            self.last_hidden_prediction = 0
        return

    #Uses Monte Carlo, in principle could be solved analitically
    def compute_threshold_for_binomial_test(self):
        #print('I am computing the threshold for the beta')
        experiments_Number = 100000
        values = np.random.binomial(self.K, self.beta, experiments_Number)
        values = np.sort(values)
        value = values[int(experiments_Number * (1 - self.inner_alpha))]
        logging.debug('binomial threshold is' + str(value))
        return value

    def update_trees(self, batch):
        new_tree = copy(self.garden[-1])
        new_tree.modify_histogram(self.batch_on_trial)
        for index in range(self.K - 1):
            self.garden[index] = copy(self.garden[index + 1])
            if self.uses_double_check:
                self.hidden_thresholds[index] = copy(self.hidden_thresholds[index + 1])
        if self.uses_double_check:
            self.hidden_thresholds[-1] = self.hidden_neural.predict_value(np.sort(new_tree.pi_values), new_tree.ndata)
            #self.hidden_thresholds[-1] = qt.ChangeDetectionTest(new_tree, self.nu, self.statistic).\
                #estimate_quanttree_threshold(self.beta, 4000)
        self.garden[-1] = new_tree
        return

    def test_batch(self, batch):
        ndata = self.garden[-1].ndata
        #number_of_points = 4000
        histogram = self.garden[-1].pi_values
        histogram = np.sort(histogram)
        threshold = self.last_neural.predict_value(histogram, ndata)
        """
        to_average = 1
        threshold = 0
        add_vector = []
        for index in range (to_average):
            to_add = qt.ChangeDetectionTest\
                (self.garden[-1],self.nu, self.statistic).estimate_quanttree_threshold(self.alpha, number_of_points)

            threshold = threshold + to_add
            add_vector.append(to_add)
        threshold = threshold/to_average
        """
        stat = self.statistic(self.garden[-1], batch)
        if stat > threshold:
             (stat, threshold)
        return stat > threshold

    def restart(self):
        self.garden =  np.empty(self.K, dtype = aux.Incremental_Quant_Tree)
        self.table = np.zeros([self.K, self.K])
        self.status = 0
        self.batch_on_trial = None
        logging.debug('I did restart')
        return

    def perform_hidden_tests(self, batch):
        """for index1 in range(self.K - 1):
            thresholds[index1] = self.hidden_neural.predict_value\
                (self.garden[index1].pi_values, self.garden[index1].ndata)
            for index2 in range(self.K - 1):
                self.table[index1][index2] = self.table[index1 + 1][index2 + 1]
        thresholds[-1] =  self.hidden_neural.predict_value\
                (self.garden[-1].pi_values, self.garden[-1].ndata)
        """
        #CODICE ALTERNATIVO IN ATTESA DELLA NN
        for index1 in range(self.K - 1):
            for index2 in range(self.K - 1):
                self.table[index1][index2] = copy(self.table[index1 + 1][index2 + 1])
        acceptances = []
        for index in range(self.K):
            acceptances.append(self.statistic(self.garden[index], batch) > self.hidden_thresholds[index])
        self.table[:, -1] = np.transpose(np.array(acceptances))
        #print ('I am hidden testing')
        return

    def secondly_refuse(self):
        value = np.sum(self.table[0])
        #logging.debug('value was' + str(value))
        return value > self.binomial_treshold

    def first_rounds(self, batch):
        if self.status == 0: #First round
            tree = aux.Incremental_Quant_Tree(initial_pi_values)
            tree.build_histogram(batch)
            self.garden[-1] = tree
        else: #Not first, but before K
            if not self.uses_double_check:
                immediately_refused = self.test_batch(batch)
                if immediately_refused:
                    self.restart()
                    return True
            tree = copy(self.garden[-1])
            tree.modify_histogram(batch)
            for index in range(self.K - 1):
                self.garden[index] = copy(self.garden[index + 1])
                if self.uses_double_check:
                    self.hidden_thresholds[index] = copy(self.hidden_thresholds[index + 1])
            if self.uses_double_check:
                self.hidden_thresholds[-1] = qt.ChangeDetectionTest(tree, self.nu, self.statistic).\
                    estimate_quanttree_threshold([0.5], 4000)
            self.garden[-1] = tree
        self.status += 1
        return False

    def play_round(self, batch):
        self.last_hidden_prediction = 0
        if self.status < self.K:
            if self.first_rounds(batch):
                return True
            return False
        immediately_refused = self.test_batch(batch)
        if self.uses_double_check:
            self.perform_hidden_tests(batch)
        if self.uses_double_check:
            refused = self.secondly_refuse()
            if refused:
                logging.debug('secondarly')
                self.last_hidden_prediction = 1
                self.restart()
                #('Secondarly')
                return True
        if immediately_refused:
            self.restart()
            logging.debug('Immediately')
            return True
        if not self.batch_on_trial is None:
            self.update_trees(self.batch_on_trial)
        self.batch_on_trial = batch
        self.status += 1
        return False

