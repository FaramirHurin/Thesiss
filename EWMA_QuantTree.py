from extendedQuantTree import Extended_Quant_Tree
import numpy as np
import extendedQuantTree as ext
import qtLibrary.libquanttree as qt



class EWMA_QuantTree:

    def __init__(self, initial_pi_values, lamb, statistic, alpha, stop, nu, desired_ARL0):
        pi_values = ext.create_bins_combination(len(initial_pi_values), 3 * len(initial_pi_values))
        self.pi_values = pi_values
        self.tree = Extended_Quant_Tree(pi_values)
        self.lamb = lamb
        self.nu = nu
        self.statistic = statistic
        self.record_history = []
        self.alpha = alpha
        self.value = self.alpha[0]
        self.beta = 1/desired_ARL0  # 1/ARL0
        self.desired_ARL0 = desired_ARL0
        self.threshold = 0
        self.max_len_computed = 0
        #self.EWMA_threshold = 0
        self.EWMA_thresholds = []
        self.stop = stop
        self.status = 0
        self.training_set = None

    def build_histogram(self, training_set, definitive = True):
        self.training_set = training_set
        if self.statistic ==  qt.pearson_statistic:
            debug = 0
        elif self.statistic == qt.tv_statistic:
            debug = 1
        else:
            raise ('Strange exception: ' + str(self.statistic))
        self.tree.build_histogram(training_set)
        if definitive:
            self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 10000)
            self.EWMA_thresholds = self.alterative_EWMA_thresholds_computation()
    """
    # MC simulation, constant threshold
    def compute_EWMA_threshold(self):
        run_lenght = 1000000
        tree, threshold, ewma_0 = self.prepare_simulated_run()
        values = self.simulate_EWMA_run(run_lenght, tree, ewma_0)
        selection = values[:int(run_lenght*self.beta)]
        return np.max(selection)
    """

    def modify_histogram(self, data, definitive = False):
        tree = self.tree
        tree.pi_values = tree.pi_values * tree.ndata
        bins = tree.find_bin(data)
        vect_to_add = np.zeros(len(tree.pi_values))
        for index in range(len(tree.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        tree.pi_values = tree.pi_values + vect_to_add
        self.tree.ndata = tree.ndata + len(data)
        self.tree.pi_values = tree.pi_values / tree.ndata
        if definitive:
            self.threshold = qt.ChangeDetectionTest(self.tree, self.nu, self.statistic).\
                estimate_quanttree_threshold(self.alpha, 10000)
            self.EWMA_thresholds = self.alterative_EWMA_thresholds_computation()
        return

    def alterative_EWMA_thresholds_computation(self):
        x = None
        max_lenght =  self.desired_ARL0 # * 3
        experiments = max_lenght * 20
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

        assert threshold > np.mean(vals) - 0.0001 #Costante necessaria per gli arrotondamenti

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
            if index % 100 == 0:
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

