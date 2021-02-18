import EWMA_QuantTree as ewma
import extendedQuantTree as ext
import neuralNetworks as NN



"""
The interface for the class is given by the init and play_round method.
During the first training_rounds the learner only trains its two components.
During the following transition_rounds both learners are used to learn, while only
the extended QuantTree is updated.
When the definitive rounds start, both are used and none is trained
"""
class Paired_Learner:

    def __init__(self, initial_pi_values, lamb, statistic,
                 alpha, nu, desired_ARL0, training_rounds, transition_rounds):
        stop = True
        self.ewma_learner = ewma.Offline_EWMA_QuantTree(initial_pi_values, lamb, statistic, alpha, stop, nu, desired_ARL0)
        self.extended_QuantTree = ext.Incremental_Quant_Tree(initial_pi_values)
        self.training_rounds = training_rounds
        self.transition_rounds = transition_rounds
        self.round = 0
        self.extended_threshold = None
        self.neural_network = NN.NN_man(len(initial_pi_values), max_N=2000, min_N= 100, nodes=100)
        self.neural_network.train(alpha)
        self.nu = nu
        self.statistic = statistic
        return

    def play_round(self, batch):
        if self.round == 0:
            self.initialize_learners(batch, [self.ewma_learner, self.extended_QuantTree])
            change = False
        elif self.round < self.training_rounds:
            definitives = [0, 0]
            change = self.play_training_round(definitives, batch)
        elif self.round == self.training_rounds:
            definitives = [1, 1]
            change = self.play_training_round(definitives, batch)
        elif self.round < self.transition_rounds:
            definitives = [0, 1]
            change = self.play_transition_round(definitives, batch)
        else:
            definitives = [0, 0]
            change = self.play_stationary_round(batch)
        self.round +=1
        return change

    #Train both EWMA and extended QuantTree
    def play_training_round(self, definitives, batch):
        self.update_learners([self.ewma_learner, self.extended_QuantTree], definitives, batch)
        return False

    #Train Extended QuantTree, use EWMA QuantTree
    def play_transition_round(self, definitives, batch):
        change = self.control([self.ewma_learner, self.extended_QuantTree], batch)
        self.update_learners([self.extended_QuantTree], definitives, batch)
        return change

    #Use Extended-QuantTree in stationary fashon
    def play_stationary_round(self, batch):
        change = self.control([self.ewma_learner, self.extended_QuantTree], batch)
        return change

    def initialize_learners(self, batch, trees):
        self.ewma_learner.build_histogram(batch, False)
        self.extended_QuantTree.build_histogram(batch)
        return

    def update_learners(self, trees, definitives, batch):
        for index in range(len(trees)):
            tree = trees[index]
            definitive = definitives[index]
            tree.modify_histogram(batch, definitive)
        if definitives[1]:
            self.update_extended_threshold()

        return

    def update_extended_threshold(self):
        self.extended_threshold = self.neural_network.predict_value\
            (self.extended_QuantTree.pi_values, self.extended_QuantTree.ndata)
        return

    def control(self, trees, batch):
        truth = 0
        for tree in trees:
            if tree == self.ewma_learner:
                truth += self.control_EWMA(batch)
            if tree == self.extended_QuantTree:
                truth += self.control_Extended_QuantTree(batch)
        return truth

    def control_EWMA(self, batch):
        self.ewma_learner.compute_EMWA(batch)
        change =  self.ewma_learner.find_change()
        return change

    def control_Extended_QuantTree(self, batch):
        stat_value = self.statistic(self.extended_QuantTree, batch)
        change = stat_value > self.extended_threshold
        return change

