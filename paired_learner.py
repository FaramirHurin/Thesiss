import EWMA_QuantTree as ewma
import extendedQuantTree as ext
import neuralNetworks as NN
import qtLibrary.libquanttree as qt

class Paired_Learner:

    def __init__(self, initial_pi_values, lamb, statistic,
                 alpha, stop, nu, desired_ARL0):
        self.ewma_learner = ewma.EWMA_QuantTree(initial_pi_values, lamb, statistic, alpha, stop, nu, desired_ARL0)
        self.extended_QuantTree = ext.Extended_Quant_Tree(initial_pi_values)
        self.training_rounds = 0
        self.transition_rounds = 0
        self.round = 0
        self.extended_threshold = 0
        self.neural_network = NN.NN_man(len(initial_pi_values), max_N=2000, min_N= 100, nodes=100)
        self.neural_network.train(alpha)
        self.nu = 0
        self.statistic = 0
        return

    def play_round(self, batch):
        if self.round < self.training_rounds:
            definitives = [0, 0]
            self.play_training_round(definitives, batch)
        if self.round == self.training_rounds:
            definitives = [1, 1]
            self.play_training_round(definitives, batch)
        elif self.round < self.transition_rounds:
            definitives = [0, 1]
            self.play_transition_round(definitives, batch)
        else:
            definitives = [0, 0]
            self.play_stationary_round(definitives, batch)
        return

    #Train both EWMA and extended QuantTree
    def play_training_round(self, definitives, batch):
        self.update_learners([self.ewma_learner, self.extended_QuantTree], definitives, batch)
        return

    #Train Extended QuantTree, use EWMA QuantTree
    def play_transition_round(self, definitives, batch):
        self.control([self.ewma_learner, self.extended_QuantTree], definitives, batch)
        self.update_learners([self.extended_QuantTree], batch)
        return

    #Use Extended-QuantTree in stationary fashon
    def play_stationary_round(self, definitives, batch):
        self.control([self.extended_QuantTree], batch)
        return

    def update_learners(self, trees, definitives, batch):
        for index in len(trees):
            tree = trees[index]
            definitive = definitives[index]
            if self.round == 0:
                tree.build_histogram(batch, definitive)
            else:
                tree.modify_histogram(batch, definitive)
        if definitives[1]:
            self.update_extended_threshold()
        return

    def update_extended_threshold(self):
        self.neural_network.predict_value\
            (self.extended_QuantTree.pi_values, self.extended_QuantTree.ndata)
        return

    def control(self, trees, batch):
        truth = 0
        for tree in trees:
            if tree == self.ewma_learner:
                truth += self.control_EWMA(batch)
            if tree == self.extended_QuantTree:
                truth += self.control_extended_QuantTree(batch)
        return

    def control_EWMA(self, batch):
        self.ewma_learner.compute_EMWA(batch)
        change =  self.ewma_learner.find_change()
        return change

    def control_Extended_QuantTree(self, batch):
        stat_value = self.statistic(self.extended_QuantTree, batch)
        change = stat_value > self.extended_threshold
        return change

