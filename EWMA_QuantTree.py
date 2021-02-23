import auxiliary_project_functions
from extendedQuantTree import Incremental_Quant_Tree
import numpy as np
import extendedQuantTree as ext
import qtLibrary.libquanttree as qt
import neuralNetworks as NN


class EWMA_QUantTree:
    def __init__(self, initial_pi_values, lamb, statistic, alpha, stop, nu, desired_ARL0):
        self.pi_values = initial_pi_values
        self.tree = Incremental_Quant_Tree(initial_pi_values)
        self.lamb = lamb
        self.nu = nu
        self.statistic = statistic
        self.record_history = []
        self.alpha = alpha
        self.value = self.alpha[0]
        self.beta = 1 / desired_ARL0  # 1/ARL0
        self.desired_ARL0 = desired_ARL0
        self.threshold = 0
        self.max_len_computed = 0
        # self.EWMA_threshold = 0
        self.EWMA_thresholds = []
        self.stop = stop
        self.status = 0
        self.training_set = None

    def build_histogram(self, training_set):
        self.training_set = training_set
        if self.statistic == qt.pearson_statistic:
            debug = 0
        elif self.statistic == qt.tv_statistic:
            debug = 1
        else:
            raise ('Strange exception: ' + str(self.statistic))
        self.tree.build_histogram(training_set)

    def modify_histogram(self, data):
        '''
         It modifies the probabilities associated to each bin according to the EXT
         tree procedure.
         Currently updating the threshold only at the last round (using MC), need to
         create an online version that uses the NN at each round
         '''
        tree = self.tree
        tree.pi_values = tree.pi_values * tree.ndata
        bins = tree.find_bin(data)
        vect_to_add = np.zeros(len(tree.pi_values))
        for index in range(len(tree.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        tree.pi_values = tree.pi_values + vect_to_add
        self.tree.ndata = tree.ndata + len(data)
        self.tree.pi_values = tree.pi_values / tree.ndata

    def alternative_EWMA_thresholds_computation(self):
        '''
        It computes the thresholds for the EWMA for max_lenght rounds.
        Multiple runs are stored.
        From the first column the highest alpha*N values are selected and the
        threshold is computed. The selected rows are then eliminated.
        :return:
        '''
        x = None
        max_lenght =  int(np.floor(self.desired_ARL0/4))
        experiments = max_lenght * 60
        table = self.fill_table(experiments, max_lenght)
        """means = np.zeros(table.shape[1])
        for index in range(len(means)):
            means[index] = np.mean(table[:,index])
        print('means are' + str(means))
        """
        self.EWMA_thresholds = self.compute_thresholds_on_table(table, None)
        self.max_len_computed = len(self.EWMA_thresholds)
        return self.EWMA_thresholds

    #Recursive calls: appends a threshold to the ones compited so far.
    #It modifies the table and passes to a new istance of itself
    def compute_thresholds_on_table(self, table, thresholds):
        if table.shape[0] < 1/self.beta or table.shape[1] < 1:
            # End Cycle
            return thresholds
        #First call
        if thresholds is None:
            thresholds = []
        values = table[:, 0]
        vals = np.sort(values)
        threshold = vals[int(len(vals) * (1 - self.beta))]
        thresholds.append(threshold)
        to_eliminate = []
        for index in range(table.shape[0]):
            if table[index, 0] > threshold:
                to_eliminate.append(index)
        table = np.delete(table, to_eliminate, axis=0)
        table = np.delete(table, 0, axis = 1)
        #Recursive call
        thresholds = self.compute_thresholds_on_table(table, thresholds)
        return thresholds

    def fill_table(self, experiments, max_lenght):
        indipendence_percentage = 0.00001
        table = np.zeros([experiments, max_lenght - 1])
        for index in range(experiments):
            if index % 500 == 0:
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
        signaled_batches = 0
        simulations_number = 4000
        for index in range(simulations_number):
            batch = np.random.uniform(0, 1, self.nu)
            positive = self.classic_batch_analysis(batch, tree)
            signaled_batches += positive
        ewma_0 = signaled_batches/simulations_number
        return tree, ewma_0

    def simulate_EWMA_run(self, max_lenght, tree, ewma_0):
        values = np.zeros(max_lenght)
        values[0] = ewma_0
        for index in range(1, max_lenght):
            batch = np.random.uniform(0, 1, self.nu)
            positive = self.classic_batch_analysis(batch, tree)
            values[index] = (1 - self.lamb) * values[index - 1] + positive * self.lamb
        return values

    def classic_batch_analysis(self, batch, tree = None):
        if tree is None:
            tree = self.tree
        stat = self.statistic(tree, batch)
        positive =  stat > self.threshold
        self.record_history.append(positive)
        return positive

    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        positive = self.classic_batch_analysis(batch, None)
        self.value = (1 - self.lamb) * self.value + positive * self.lamb
        self.status += 1
        return self.value

    def find_change(self):
        # If there is a change we return True
        index = min(len(self.EWMA_thresholds) - 1, self.status)
        try:
            change = self.value > self.EWMA_thresholds[index]
        except:
            print(len(self.EWMA_thresholds), index)
            raise Exception
        return change

    def play_round(self, batch):
        self.compute_EMWA(batch)
        change = self.find_change()
        return change

class Offline_EWMA_QuantTree(EWMA_QUantTree):

    def build_histogram(self, training_set, definitive = True):
        super().build_histogram(training_set)
        if definitive:
            self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 10000)
            self.EWMA_thresholds = self.alternative_EWMA_thresholds_computation()

    def modify_histogram(self, data, definitive = False):
        super().modify_histogram(data, definitive)
        if definitive:
            self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 10000)
            self.EWMA_thresholds = self.alternative_EWMA_thresholds_computation()
        return

class Online_EWMA_QUantTree(EWMA_QUantTree):
    def __init__(self, initial_pi_values, lamb, statistic, alpha, stop, nu, desired_ARL0):
        super().__init__(initial_pi_values, lamb, statistic, alpha, stop, nu, desired_ARL0)
        bins_number = len(initial_pi_values)
        self.neural_network = NN.NN_man(bins_number, bins_number * 200, bins_number * 2, 10)
        self.neural_network.train(self.alpha)
        self.buffer = None

    def build_histogram(self, training_set):
        super().build_histogram(training_set)
        self.threshold = self.neural_network.predict_value(self.tree.pi_values, self.tree.ndata)
        self.EWMA_thresholds = self.alternative_EWMA_thresholds_computation()

    def modify_histogram(self, data):
        super().modify_histogram(data)
        self.threshold = self.neural_network.predict_value(self.tree.pi_values, self.tree.ndata)

    def play_round(self, batch):
        change = super().play_round(batch)
        if not change:
            if self.buffer is not None:
                self.modify_histogram(self.buffer)
            self.buffer = batch
        return change