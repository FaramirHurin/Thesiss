from main_code.incremental_QuantTree import Incremental_Quant_Tree
import numpy as np
import qtLibrary.libquanttree as qt
from main_code import neuralNetworks as NN


class EWMA_QUantTree:
    def __init__(self, initial_pi_values, lamb, statistic, alpha, nu, desired_ARL0):
        self.pi_values = initial_pi_values
        self.tree = Incremental_Quant_Tree(initial_pi_values)
        self.lamb = lamb
        self.nu = nu
        self.statistic = statistic
        self.record_history = []
        self.alpha = alpha
        self.beta = 1 / desired_ARL0  # 1/ARL0
        self.desired_ARL0 = desired_ARL0
        self.threshold = None
        self.max_len_computed = None
        self.EWMA_thresholds = []
        self.status = 0
        self.values = [alpha[0]]
        self.training_set = None
        self.change_round = None

    def build_histogram(self, training_set, definitive = True):
        self.training_set = training_set
        if self.statistic == qt.pearson_statistic:
            debug = 0
        elif self.statistic == qt.tv_statistic:
            debug = 1
        else:
            raise ('Strange exception: ' + str(self.statistic))
        self.tree.build_histogram(training_set)
        try:
            if self.threshold:
                debug = 0
        except:
            raise (Exception)
        self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 40000)
        self.alternative_EWMA_thresholds_computation()
        return

    def modify_histogram(self, data):
        tree = self.tree
        tree.pi_values = tree.pi_values * tree.ndata
        bins = tree.find_bin(data)
        vect_to_add = np.zeros(len(tree.pi_values))
        for index in range(len(tree.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        tree.pi_values = tree.pi_values + vect_to_add
        self.tree.ndata = tree.ndata + len(data)
        self.tree.pi_values = tree.pi_values / tree.ndata
        return

    def alternative_EWMA_thresholds_computation(self):
        '''
        It computes the thresholds for the EWMA for max_lenght rounds.
        Multiple runs are stored.
        From the first column the highest alpha*N values are selected and the
        threshold is computed. The selected rows are then eliminated.
        :return:
        '''
        x = None
        max_lenght =  int(self.desired_ARL0/3)
        self.max_len_computed = max_lenght - 1
        experiments = max_lenght * 200
        table = self.fill_table(experiments, max_lenght)
        self.compute_thresholds_on_table(table)
        return

    #Recursive calls: appends a threshold to the ones compited so far.
    #It modifies the table and passes to a new istance of itself
    def compute_thresholds_on_table(self, table):
        if table.shape[0] < 1/self.beta or table.shape[1] < 1:
            # End Cycle
            return
        values = table[:, 0]
        vals = np.sort(values)
        threshold = vals[int(len(vals) * (1 - self.beta))]
        self.EWMA_thresholds.append(threshold)
        to_eliminate = []
        for index in range(table.shape[0]):
            if table[index, 0] > threshold:
                to_eliminate.append(index)
        table = np.delete(table, to_eliminate, axis=0)
        table = np.delete(table, 0, axis = 1)
        #Recursive call
        self.compute_thresholds_on_table(table)
        return

    def fill_table(self, experiments, max_lenght):
        #Indipendence percentage represents the number of different trees used for threshold computation.
        #Used to average over the noise on trees generation
        indipendence_percentage = 1/1000
        table = np.zeros([experiments, max_lenght])
        tree = None
        for index in range(experiments):
            if index * indipendence_percentage %1 == 0:
                tree = self.prepare_simulated_run()
            values = self.simulate_EWMA_run(max_lenght, tree)
            table[index] = values
        return table

    def prepare_simulated_run(self):
        training_set = np.random.uniform(0, 1, self.tree.ndata)
        tree = qt.QuantTreeUnivariate(self.tree.pi_values)
        tree.build_histogram(training_set)
        return tree

    def simulate_EWMA_run(self, max_lenght, tree):
        values = np.zeros(max_lenght)
        values[0] = self.values[0]
        for index in range(1, max_lenght):
            batch = np.random.uniform(0, 1, self.nu)
            positive = self.classic_batch_analysis(batch, tree)
            try:
                values[index] = (1 - self.lamb) * values[index - 1] + positive * self.lamb
            except:
                raise(Exception)
        return values

    def classic_batch_analysis(self, batch, tree = None):
        if tree is None:
            tree = self.tree
        stat = self.statistic(tree, batch)
        positive =  stat > self.threshold
        self.record_history.append(positive)
        try:
            if positive:
                debug = 0
        except:
            raise Exception
        return positive

    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        positive = self.classic_batch_analysis(batch, None)
        self.status += 1
        self.values.append((1 - self.lamb) * self.values[-1] + positive * self.lamb)
        return

    def find_change(self):
        # If there is a change we return True
        index = min(self.max_len_computed, self.status)
        #print('Index is' + str(index))
        try:
            if self.values[-1] > self.EWMA_thresholds[index]:
                self.change_round = self.status
                return True
        except:
            raise Exception #TODO Self.EWMAThresholds are None here for online QT
        return False

    def play_round(self, batch):
        #assert self.change_round is None
        self.compute_EMWA(batch)
        change = self.find_change()
        if self.change_round is not None:
            return True
        else:
            return False

    def restart(self):
        self.values = [self.alpha[0]]
        # assert self.values is not None
        self.status = 0
        self.change_round = None
        return

class Offline_EWMA_QuantTree(EWMA_QUantTree):

    def modify_histogram(self, data, definitive = False):
        super().modify_histogram(data, definitive)
        if definitive:
            self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 40000)
            try:
                if self.threshold:
                    debug = 0
            except:
                raise(Exception)
            self.EWMA_thresholds = self.alternative_EWMA_thresholds_computation()
        return

class Online_EWMA_QUantTree(EWMA_QUantTree):
    def __init__(self, initial_pi_values, lamb, statistic, alpha, nu, desired_ARL0):
        super().__init__(initial_pi_values, lamb, statistic, alpha, nu, desired_ARL0)
        bins_number = len(initial_pi_values)
        self.neural_network = NN.NN_man(bins_number, bins_number * 200, bins_number * 2, 10)
        self.neural_network.train(self.alpha)
        self.buffer = None

    def modify_histogram(self, data):
        super().modify_histogram(data)
        self.threshold = self.neural_network.predict_value(self.tree.pi_values, self.tree.ndata)
        try:
            if self.threshold:
                debug = 0
        except:
            raise (Exception)

    def play_round(self, batch):
        change = super().play_round(batch)
        if not change:
            if self.buffer is not None:
                self.modify_histogram(self.buffer)
            self.buffer = batch
        return change