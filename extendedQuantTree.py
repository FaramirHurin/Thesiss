import numpy as np
import qtLibrary.libquanttree as qt
import neuralNetworks as nn

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
        self.network = nn.NN_man()
        self.network.train(alpha)
        self.buffer = None
        ìthreshold = None
        self.statistic = statistic
        return

    def build_histogram(self, data):
        self.tree.build_histogram(data)

    def play_round(self, batch):
        threshold = self.network.predict_value(self.tree.pi_values, self.tree.leaves)
        change = self.statistic(self.tree, batch) > threshold
        if not change:
            if self.buffer:
                self.tree.modify_histogram(self.buffer)
            self.buffer = batch
        return change


