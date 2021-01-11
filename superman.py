import qtLibrary.libquanttree as qt
from extendedQuantTree import Data_set_Handler, Extended_Quant_Tree, Alternative_threshold_computation
from neuralNetworks import NN_man

#Main interface to use and run experiments on extended QuantTree
class Superman:

    def __init__(self, percentage, SKL, initial_pi_values, data_number,
                 alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                 data_number_for_learner):
        self.data_number = data_number
        self.bins_number = bins_number
        self.data_dimension = data_Dimension
        self.nu = nu
        self.B = B
        self.statistic = statistic
        self.max_N = max_N
        self.data_number_for_learner = data_number_for_learner
        self.alpha = alpha
        self.initial_pi_values = initial_pi_values
        self.SKL = SKL
        self.percentage = percentage
        self.handler = Data_set_Handler(self.data_dimension)

    def create_training_set_for_QT(self):
        self.data_set = self.handler.return_equal_batch(self.data_number)
        return

    def run_normal_algorithm(self, number_of_experiments, equal = True):
        tree = qt.QuantTree(self.initial_pi_values)
        tree.build_histogram(self.data_set)
        assert (tree.ndata == len(self.data_set))
        test = qt.ChangeDetectionTest(tree, self.nu, self.statistic)
        thr = test.estimate_quanttree_threshold(self.alpha, self.B)
        value = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.return_equal_batch(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        return value / number_of_experiments

    def run_modified_algorithm_without_learner(self, number_of_experiments, equal = True):
        initial_db_size = int(len(self.data_set)*self.percentage)
        initial_data = self.data_set[0:initial_db_size]
        sequent_data = self.data_set[initial_db_size:len(self.data_set)]
        tree = Extended_Quant_Tree(self.initial_pi_values)
        tree.build_histogram(initial_data)
        tree.modify_histogram(sequent_data)
        test = qt.ChangeDetectionTest(tree, self.nu, self.statistic)
        thr = test.estimate_quanttree_threshold(self.alpha, self.B)

        tree2 = qt.QuantTree(tree.pi_values)
        tree2.build_histogram(self.data_set)
        test2 = qt.ChangeDetectionTest(tree2, self.nu, self.statistic)
        thr2 = test.estimate_quanttree_threshold(self.alpha, self.B)
        value = 0
        value2 = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.return_equal_batch(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            y2 = self.statistic(tree2, batch)
            value += y > thr
            value2 += y2 > thr2

        return value/number_of_experiments, value2/number_of_experiments

    def run_asymtpotic_algorithm_without_learner(self, number_of_experiments, equal = True):
        initial_db_size = int(len(self.data_set) * self.percentage)
        initial_data = self.data_set[0:initial_db_size]
        sequent_data = self.data_set[initial_db_size:len(self.data_set)]
        tree = Extended_Quant_Tree(self.initial_pi_values)
        tree.build_histogram(initial_data)
        tree.modify_histogram(sequent_data)
        test = Alternative_threshold_computation(tree.pi_values, self.nu, self.statistic)
        thr = test.compute_threshold(self.alpha, self.B)
        value = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.return_equal_batch(self.nu)
            else:
                batch = self.handler.return_equal_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr

        return value / number_of_experiments


        #Uses as a prediction for the threshold the average of the thesholds in the DataBase

    def run_Dummy(self, number_of_experiments, equal = True):
        initial_db_size = int(len(self.data_set) * self.percentage)
        initial_data = self.data_set[0:initial_db_size]
        sequent_data = self.data_set[initial_db_size:len(self.data_set)]
        tree = Extended_Quant_Tree(self.initial_pi_values)
        tree.build_histogram(initial_data)
        tree.modify_histogram(sequent_data)
        man = NN_man(self.bins_number, self.max_N, 30, 10)
        thr = man.compute_dummy_prediction()
        value = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.return_equal_batch(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        if equal:
            return (number_of_experiments - value) / number_of_experiments
        else:
            return value / number_of_experiments

    #TODO move here the SKL part
    def run_modified_algorithm_with_learner(self, number_of_experiments, min_N, alpha, equal = True):
        initial_db_size = int(len(self.data_set) * self.percentage)
        initial_data = self.data_set[0:initial_db_size]
        sequent_data = self.data_set[initial_db_size:len(self.data_set)]
        tree = Extended_Quant_Tree(self.initial_pi_values)
        tree.build_histogram(initial_data)
        tree.modify_histogram(sequent_data)
        man = NN_man(50, self.max_N, 30, 200)
        man.train(alpha)
        thr = man.predict_value(tree.pi_values, tree.ndata)
        value = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.return_equal_batch(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr

        return value / number_of_experiments