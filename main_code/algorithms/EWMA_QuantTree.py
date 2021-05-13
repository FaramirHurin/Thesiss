from main_code.algorithms.incremental_QuantTree import Incremental_Quant_Tree
import numpy as np
import qtLibrary.libquanttree as qt
from main_code import neuralNetworks as NN
import matplotlib.pyplot as plt

MAX_LENGHT = 25
EXPERIMENTS = MAX_LENGHT * 1200

#Copied from paper
TABLE = {
    100: {
        0: 2.76,
        1: -6.23,
        3: 18.12,
        5: -312.45,
        7: 1002.18
    },
    400: {
        0: 3.76,
        1: -6.23,
        3: 18.12,
        5: -330.13,
        7: 848.18
    }
}


class EWMA_QUantTree:
    def __init__(self, initial_pi_values, lamb, statistic, alpha, nu, desired_ARL0):
        self.pi_values = initial_pi_values
        self.tree = Incremental_Quant_Tree(initial_pi_values)
        self.lamb = lamb
        self.nu = nu
        self.statistic = statistic
        self.alpha = alpha
        self.beta = 1 / desired_ARL0  # 1/ARL0
        self.desired_ARL0 = desired_ARL0

        #self.EWMA_thresholds = ewma_thresholds
        self.FPR_estimator = [0]
        self.threshold = None
        self.record_history = []
        self.status = 1
        self.values = [0]
        self.training_set = None
        self.change_round = None
        self.tie_breaker = 0
        self.thresholds = [0]

    #Must be called from outside. Here the EWMA thresholds are initialized if not called from outside
    def build_histogram(self, training_set, definitive = True):
        self.training_set = training_set
        self.tree.build_histogram(training_set)
        #assert self.threshold is not  None
        self.threshold, self.tie_breaker = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 6000, True)
        return

    def modify_histogram(self, data):
        tree = self.tree
        tree.modify_histogram(data)
        """
        tree.pi_values = tree.pi_values * tree.ndata
        bins = tree.find_bin(data)
        vect_to_add = np.zeros(len(tree.pi_values))
        for index in range(len(tree.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        tree.pi_values = tree.pi_values + vect_to_add
        self.tree.ndata = tree.ndata + len(data)
        self.tree.pi_values = tree.pi_values / tree.ndata
        """
        return
    """
    #Returns self.EWMA_values. Computes the method descripted in Ross
    @classmethod
    def ross_simulation(cls):

        EWMA_Thresholds = []
        TABLE = 0 #TODO Read from TABLE
        if TABLE is not None:
            EWMA_Thresholds = TABLE
        else:
            ARLO = [100]
            p0 = range(start=0.01, stop=0.99, range=98)
            L_table = np.zeros([len(ARLO), len(p0)])
            for arl in range(ARLO):
                for p in range(p0):
                    L_table[arl, p] = cls.find_L_given_p_and_ARLO(p, arl)
            cls.fit_regressor()
            deubug = 0 #TODO Simulations and TABLE storage
        return EWMA_Thresholds

    @classmethod
    def find_L_given_p_and_ARLO(cls, p, ARLO):
        L = 0
        return L

    @classmethod
    def fit_regressor(cls):
        return
    """

    def classic_batch_analysis(self, batch, tree = None):
        if tree is None:
            tree = self.tree
        stat = self.statistic(tree, batch)
        positive =  stat > self.threshold
        if not (positive == True or positive == False):
            positive = positive[0]
        """head_tail = np.random.binomial(1, self.tie_breaker)
        if stat == self.threshold and head_tail == 1:
            positive = True
        self.record_history.append(positive)
        """
        assert positive == True or positive == False
        self.record_history.append(positive)
        return positive

    # Compute current value of EMWA based on the value at previous round.
    # Important: it updates the attributes of the object
    def compute_EMWA(self, batch):
        positive = self.classic_batch_analysis(batch, None)
        self.status += 1
        value_to_append = self.values[-1] * (1 - self.lamb) + positive * self.lamb

        self.values.append(value_to_append)
        to_add = (self.FPR_estimator[-1] * (self.status - 1) + positive) / self.status
        self.FPR_estimator.append(to_add)
        return

    def find_change(self):
        # If there is a change we return True
        #print('Index is' + str(index))
        confront = self.compute_thr()
        try:
            if self.values[-1] > confront:
                self.change_round = self.status
                return True
        except:
            raise Exception
        return False

    def compute_thr(self):
        my_table = TABLE[self.desired_ARL0]
        L = 0
        for key in my_table.keys():
            L += my_table[key] * (self.FPR_estimator[-1] ** key)
        L = max(0, L)
        sigma_pi = self.FPR_estimator[-1] * (1 - self.FPR_estimator[-1])
        assert sigma_pi >= 0
        sigma_eta = sigma_pi * np.sqrt((self.lamb)/(2 - self.lamb)*(1 -(1 - self.lamb)**(2*self.status)))
        assert sigma_eta >= 0 # and sigma_eta < 1
        if L <0:
            print(L)
        to_return = self.FPR_estimator[-1] + L * sigma_eta
        self.thresholds.append(to_return)
        #print(to_return)
        return to_return

    def play_round(self, batch):
        assert len(self.values) == len(self.FPR_estimator)
        #assert self.change_round is None
        self.compute_EMWA(batch)
        change = self.find_change()
        """if self.status == 500:
            plt.plot(self.values)
            plt.plot(self.FPR_estimator)
            plt.plot(self.thresholds)
            plt.title('Values and estimators')
            plt.show()
        """
        return change

    def restart(self):
        self.values = [0]
        self.FPR_estimator = [0]
        self.thresholds = [0]
        assert self.values is not None
        self.status = 0
        self.change_round = None
        return

class Offline_EWMA_QuantTree(EWMA_QUantTree):

    def modify_histogram(self, data, definitive = False):
        super().modify_histogram(data)
        if definitive:
            self.threshold, self.tie_breaker = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 40000, True)
            assert self.threshold
            #self.EWMA_thresholds = self.alternative_EWMA_thresholds_computation()
        return

class Online_EWMA_QUantTree(EWMA_QUantTree):
    def __init__(self, initial_pi_values, lamb, statistic, alpha, nu, desired_ARL0):
        super().__init__(initial_pi_values, lamb, statistic, alpha, nu, desired_ARL0)
        bins_number = len(initial_pi_values)
        self.neural_network = NN.NN_man(bins_number, bins_number * 200, bins_number * 2, 30)
        self.neural_network.train(self.alpha)
        self.buffer = None

    def modify_histogram(self, data):
        super().modify_histogram(data)
        self.threshold = self.neural_network.predict_value(self.tree.pi_values, self.tree.ndata)
        assert self.threshold
        return

    def play_round(self, batch):
        change = super().play_round(batch)
        if not change:
            if self.buffer is not None:
                self.modify_histogram(self.buffer)
            self.buffer = batch
        """
        if self.status == 50:
            plt.plot(self.values)
            #plt.plot(self.FPR_estimator)
            plt.plot(self.thresholds)
            plt.title('Values and estimators')
            plt.show()
        """
        return change




"""
PREDECENTE CALCOLO DELLE SOGLIE
    def alternative_EWMA_thresholds_computation(self):
        '''
        It computes the thresholds for the EWMA for max_lenght rounds.
        Multiple runs are stored.
        From the first column the highest alpha*N values are selected and the
        threshold is computed. The selected rows are then eliminated.
        :return:
        '''
        #print('Strano')
        self.max_len_computed = MAX_LENGHT - 1
        TABLE = self.fill_table(EXPERIMENTS, MAX_LENGHT)
        self.compute_thresholds_on_table(TABLE)
        return

    #Recursive calls: appends a threshold to the ones compited so far.
    #It modifies the TABLE and passes to a new istance of itself
    def compute_thresholds_on_table(self, TABLE):
        if TABLE.shape[0] <  1/self.beta or TABLE.shape[1] < 1:
            # End Cycle
            return
        #First cycle initialization
        if self.EWMA_thresholds is None:
            self.EWMA_thresholds = []
        #Normal run
        values = TABLE[:, 0]
        vals = np.sort(values)
        threshold = vals[int(len(vals) * (1 - self.beta))]
        self.EWMA_thresholds.append(threshold)
        to_eliminate = []
        for index in range(TABLE.shape[0]):
            if TABLE[index, 0] > threshold:
                to_eliminate.append(index)
        TABLE = np.delete(TABLE, to_eliminate, axis=0)
        TABLE = np.delete(TABLE, 0, axis = 1)
        #Recursive call
        self.compute_thresholds_on_table(TABLE)
        return

    def fill_table(self, experiments, max_lenght):
        #Indipendence percentage represents the number of different trees used for threshold computation.
        #Used to average over the noise on trees generation
        indipendence_percentage = 1/2000
        TABLE = np.zeros([experiments, max_lenght])
        tree = None
        for index in range(experiments):
            if index % 1000 == 0:
                print (index)
            if index * indipendence_percentage %1 == 0:
                tree = self.prepare_tree_for_simulation()
            values = self.simulate_EWMA_run(max_lenght, tree)
            TABLE[index] = values
        return TABLE

    def prepare_tree_for_simulation(self):
        training_set = np.random.uniform(0, 1, self.tree.ndata)
        tree = qt.QuantTreeUnivariate(self.tree.pi_values)
        tree.build_histogram(training_set)
        return tree

    def simulate_EWMA_run(self, max_lenght, tree):
        values = np.zeros(max_lenght)
        values[0] = self.values[0]
        for index in range(1, max_lenght):
            batch = np.random.uniform(0, 1, self.NU)
            positive = self.classic_batch_analysis(batch, tree)
            try:
                values[index] = (1 - self.lamb) * values[index - 1] + positive * self.lamb
            except:
                raise(Exception)
        return values
"""
