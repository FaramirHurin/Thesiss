import main_code.algorithms.incremental_QuantTree as iqt
import numpy as np
import matplotlib as plt
import qtLibrary.libquanttree as qt
import logging
logger = logging.getLogger('logger')


class EWMA_tree:
    def __init__(self, alphaIQT, lambd,
                 desired_ARL0, initial_pi_values, statistic, training_set):
        self.values = None
        self.tree = iqt.Online_Incremental_QuantTree(initial_pi_values, alphaIQT, statistic)
        self.tree.build_histogram(training_set)
        self.lambd = lambd
        self.EWMA = alphaIQT
        self.EWMA_thresholds = None
        self.compute_thresholds(desired_ARL0)
        self.time = 0
        return

    def compute_thresholds(self, desired_ARL0):
        experiments = 1000
        max_lenght = 4 * desired_ARL0
        table = self.fill_table(experiments, max_lenght)
        self.compute_thresholds_on_table(table)
        return

    def play_round(self, batch):
        tree_outcome = self.tree.play_round(batch)
        self.compute_EWMA(tree_outcome)
        stop = self.does_stop()
        self.time += 1
        return stop

    def compute_EWMA(self, tree_outcome):
        self.EWMA = self.lambd * tree_outcome + (1 - self.lambd) * self.EWMA
        return self.EWMA

    def does_stop(self):
        return self.EWMA > self.thresholds[self.time]

    #Recursive calls: appends a threshold to the ones compited so far.
    #It modifies the TABLE and passes to a new istance of itself
    def compute_thresholds_on_table(self, TABLE):
        while not TABLE.shape[0] <  1/self.beta or TABLE.shape[1] < 1:
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
        return

    def fill_table(self, experiments, max_lenght):
        #Indipendence percentage represents the number of different trees used for threshold computation.
        #Used to average over the noise on trees generation
        indipendence_percentage = 1/2000
        TABLE = np.zeros([experiments, max_lenght])
        tree = None
        for index in range(experiments):
            logger.debug('Experiment number ' + str(index))
            if index % 1000 == 0:
                print (index)
            if index * indipendence_percentage %1 == 0:
                tree = self.prepare_tree_for_simulation()
            values = self.simulate_EWMA_run(max_lenght, tree)
            TABLE[index] = values
        return TABLE

    def prepare_tree_for_simulation(self):
        training_set = np.random.uniform(0, 1, self.tree.tree.ndata)
        tree = qt.QuantTreeUnivariate(self.tree.tree.pi_values)
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
    """

