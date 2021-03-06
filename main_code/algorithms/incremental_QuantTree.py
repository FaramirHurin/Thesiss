import numpy as np
import qtLibrary.libquanttree as qt
from main_code import neuralNetworks as nn


# Uses cuts on space instead of the ones on probabilites, equal to normal cut with N=Inf

# Extends canonic quantTree with the possibility to modify the histogram associated
class Incremental_Quant_Tree(qt.QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)
        self.ndata = 0

    def build_histogram(self, data, do_PCA=False):
        super().build_histogram(data, do_PCA)
        self.ndata = len(data)

    def modify_histogram(self, data, definitive = True):
        self.pi_values = self.pi_values * self.ndata
        bins = self.find_bin(data)
        vect_to_add = np.zeros(len(self.pi_values))
        for index in range(len(self.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        self.pi_values = self.pi_values + vect_to_add
        self.ndata = self.ndata + len(data)
        self.pi_values = self.pi_values / self.ndata
        return



class Online_Incremental_QuantTree:
    def __init__(self, pi_values, alpha, statistic):
        self.tree = Incremental_Quant_Tree(pi_values)
        bins_number = len(pi_values)
        self.alpha = alpha
        self.statistic = statistic

        self.network = nn.NN_man(bins_number, 200 * bins_number, 3 * bins_number, 30)
        self.network.train(alpha)

        self.buffer = None
        self.change_round = None
        self.round = 0
        return

    def build_histogram(self, data):
        self.tree.build_histogram(data)

    def play_round(self, batch):
        threshold = self.network.predict_value(self.tree.pi_values, self.tree.ndata)
        #threshold2 = qt.ChangeDetectionTest(self.tree, len(batch), self.statistic).estimate_quanttree_threshold(self.alpha, 4000)
        stat = self.statistic(self.tree, batch)
        change = stat > threshold
        if not change:
            self.update_model(batch)
        else:
            self.change_round = self.round
        self.round +=1
        return change

    def restart(self):
        self.round = 0
        self.change_round = None
        self.buffer = None

    def update_model(self, batch):
        if self.buffer is not None:
            self.tree.modify_histogram(self.buffer)
        self.buffer = batch
        return




