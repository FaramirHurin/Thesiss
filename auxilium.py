import numpy as np
import libquanttree as qt
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neural_network as nn
import libccm as ccm


# Creates and handles the dataset of the pi_values combinations
class DataSet_for_the_learner:

    def __init__(self, bins_numer, data_number):
        self.bins_number = bins_numer
        self.data_number = data_number
        self.histograms = np.zeros((data_number, bins_numer))
        self.thresholds = {}
        self.asymptotyical_thresholds = np.zeros(data_number)
        return

    def create_multiple_bins_combinations(self):
        for histogram_number in range(self.data_number):
            self.histograms[histogram_number] = \
                create_bins_combination(self.bins_number)
        return self.histograms

    def associate_thresholds(self, N, nu, stat, alpha, B):
        self.thresholds[N] = np.zeros(self.data_number)
        for histogram_number in range(self.data_number):
            if histogram_number % 10 == 0:
                print ('histogram number '+str(histogram_number))
            tree = qt.QuantTree(self.histograms[histogram_number])
            tree.ndata = N
            test = qt.ChangeDetectionTest(tree, nu, stat)
            threshold = test.estimate_quanttree_threshold(alpha, B)
            self.thresholds[N][histogram_number] = threshold
        return self.thresholds

    def asymptotically_associate_thresholds(self, nu, stat, alpha, B):
        for histogram_number in range(self.data_number):
            alternative_computation = \
                Alternative_threshold_computation(self.histograms[histogram_number], nu, stat)
            threshold = alternative_computation.compute_threshold(alpha, B)
            self.asymptotyical_thresholds[histogram_number] = threshold
        return self.asymptotyical_thresholds


# Uses cuts on space instead of the ones on probabilites, equal to normal cut with N=Inf
class Alternative_threshold_computation:

    def __init__(self, pi_values, nu, statistic):
        self.pi_values = pi_values
        self.nu = nu
        self.statistic = statistic
        return

    # Versione modificata di qt.QuantTreeUnivariate.buildHistogram()
    # TODO Rivedere logica leaves
    def compute_cut(self):
        definitive_pi_values = np.zeros(len(self.pi_values))
        histogram = np.zeros(len(self.pi_values)+1)
        bins = []
        interval_still_to_cut = [0, 1]
        left_count = 1
        right_count = 1
        for value in self.pi_values:
            bernoulli_value = np.random.binomial(1, 0.5)
            if bernoulli_value == 0:
                interval_still_to_cut[0] = interval_still_to_cut[0] + value
                histogram[left_count] = interval_still_to_cut[0]
                definitive_pi_values[left_count-1] = value
                left_count +=1
            else:
                interval_still_to_cut[1] = interval_still_to_cut[1] - value
                histogram[-right_count-1] = interval_still_to_cut[1]
                definitive_pi_values[- right_count] = value
                right_count += 1

        histogram = np.transpose(histogram)
        histogram[0] = 0
        histogram[-1] = 1
        self.pi_values = definitive_pi_values
        tree = qt.QuantTreeUnivariate(self.pi_values)
        tree.leaves = histogram
        return tree

    def compute_threshold(self, alpha, B):
        alpha = alpha[0]
        stats = []
        histogram = self.compute_cut()
        for b_count in range(B):
            W = np.random.uniform(0, 1, self.nu)
            thr = self.statistic(histogram, W)
            stats.append(thr)
        stats.sort()
        threshold = stats[int((1-alpha)*B)]
        return threshold


# The neural network used to guess the thresholds.
# For small N we will average
class Learner:

    def __init__(self, bins_number, statistic, N_values):
        layers_size = 4
        self.bins_number = bins_number
        self.statistic = statistic
        self.N_values = N_values
        self.nets = {}
        self.asymptpotic_net = nn.MLPRegressor(layers_size, solver = 'lbfgs', verbose = False, learning_rate='adaptive')
        # TODO Set hyperparameters
        self.max_N = max(N_values)
        for index in self.N_values:
            self.nets[index] = nn.MLPRegressor(layers_size, solver = 'lbfgs', verbose = False, learning_rate='adaptive', alpha = 0.0005)
        return

    def tune_model_for_a_given_N(self, histograms, thresholds, N):
        self.nets[N].fit(histograms, thresholds)
        return

    def tune_model_for_asymptotical_values(self, histograms, thresholds):
        self.asymptpotic_net.fit(histograms, thresholds)
        return

    def predict_threshold(self, histogram, N):
        histogram = np.sort(histogram)
        if N in self.nets.keys():
            return self.nets[N].predict(histogram.reshape(1, -1))
        elif N < min((self.nets.keys())):
            return self.nets[min(self.nets.keys())].predict(histogram.reshape(1, -1))
        elif N < self.max_N:
            biggest_Smaller = 0
            smallest_Bigger = self.max_N
            for elem in self.nets.keys():
                if elem < N and elem > biggest_Smaller:
                    biggest_Smaller = elem
                if elem > N and elem < smallest_Bigger:
                    smallest_Bigger = elem
            small_prediction = \
                self.nets[biggest_Smaller].predict(histogram.reshape(1, -1))
            big_prediction = \
                self.nets[smallest_Bigger].predict(histogram.reshape(1, -1))
            prediction = small_prediction + (big_prediction - small_prediction) * \
                         ((N - biggest_Smaller) / (smallest_Bigger - biggest_Smaller))
            return prediction
        else:
            print ("I am asymptotic")
            return self.asymptpotic_net.predict(histogram.reshape(1, -1))


# Extends canonic quantTree with the possibilityu to modify the histogram associated
class Extended_Quant_Tree(qt.QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)
        self.ndata = 0

    def build_histogram(self, data, do_PCA=False):
        super().build_histogram(data, do_PCA)
        self.ndata = len(data)

    def modify_histogram(self, data):
        self.pi_values = self.pi_values * self.ndata
        bins = self.find_bin(data)
        vect_to_add = np.zeros(len(self.pi_values))
        for index in range(len(self.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        self.pi_values = self.pi_values + vect_to_add
        self.ndata = self.ndata + len(data)
        self.pi_values = self.pi_values / self.ndata
        return


class Data_set_Handler:

    def __init__(self, dimensions_number):
        self.dimensions_number = dimensions_number
        self.gauss0 = ccm.random_gaussian(self.dimensions_number)
        return

    def generate_data_set(self, data_number):
        self.data_set = np.random.multivariate_normal(self.gauss0[0], self.gauss0[1], data_number)
        return self.data_set

    def return_equal_batch(self, B):
        return np.random.multivariate_normal(self.gauss0[0], self.gauss0[1], B)

    def generate_similar_batch(self, B, target_sKL):
        rot, shift = ccm.compute_roto_translation(self.gauss0, target_sKL)
        gauss1 = ccm.rotate_and_shift_gaussian(self.gauss0, rot, shift)
        return np.random.multivariate_normal(gauss1[0], gauss1[1], B)

#TODO Modify in order to be used for different alpha (here or in the regressor)
class Regressed_Change_Detection_Test:
    def __init__(self, model, ndata, regressor):
        self.model = model
        self.threshold = regressor.predict_threshold(model.pi_values, ndata)

    def reject_null_hypothesis(self, W, alpha, statistic):
         y = statistic(self.model, W)
         return y > self.threshold



class Superman:
    def __init__(self, percentage, SKL, initial_pi_values, data_number,
                 alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                 data_number_for_learner):
        self.data_number = data_number
        self.bins_number = bins_number
        self.data_dimension = data_Dimension
        self.nu = nu
        self.B = B
        self.statistic = statistic
        self.N_values = N_values
        self.data_number_for_learner = data_number_for_learner
        self.alpha = alpha
        self.initial_pi_values = initial_pi_values
        self.SKL = SKL
        self.percentage = percentage
        self.learner = None
        self.handler = Data_set_Handler(self.data_dimension)

    #N_to_train are the values of N for the dictionary
    def create_and_train_net(self):
        print('Training')
        self.learner = Learner(self.bins_number, self.statistic, self.N_values)
        data_for_learner = DataSet_for_the_learner(self.bins_number,  self.data_number_for_learner)
        data_for_learner.create_multiple_bins_combinations()
        histograms = data_for_learner.histograms
        for value in self.N_values:
            print('training ' + str(value))
            thresholds = data_for_learner.associate_thresholds(value, self.nu, self.statistic, self.alpha, self.B)
            self.learner.tune_model_for_a_given_N(histograms, thresholds[value], value)
        asymptotic_thresholds = []
        for hist in histograms:
            asymptotic_computer = Alternative_threshold_computation(hist, self.nu, self.statistic)
            asymptotic_thresholds.append(asymptotic_computer.compute_threshold(self.alpha, self.B))
        self.learner.tune_model_for_asymptotical_values(histograms, asymptotic_thresholds)
        return

    def create_training_set_for_QT(self):
        self.data_set = self.handler.generate_data_set(self.data_number)
        return

    def run_normal_algorithm(self, number_of_experiments, equal = True):
        tree = qt.QuantTree(self.initial_pi_values)
        tree.build_histogram(self.data_set)
        assert (tree.ndata == len(self.data_set))
        test = qt.ChangeDetectionTest(tree, self.nu, self.statistic)
        test.statistic = qt.pearson_statistic
        thr = test.estimate_quanttree_threshold(self.alpha, self.B)
        value = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        return (number_of_experiments - value) / number_of_experiments

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
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            y2 = self.statistic(tree2, batch)
            value += y > thr
            value2 += y2 > thr2
        return ( number_of_experiments - value)/number_of_experiments, ( number_of_experiments - value2)/number_of_experiments

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
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        return (number_of_experiments - value) / number_of_experiments

    #TODO move here the SKL part
    def run_modified_algorithm_with_learner(self, number_of_experiments, equal = True):
        self.create_and_train_net()
        initial_db_size = int(len(self.data_set) * self.percentage)
        initial_data = self.data_set[0:initial_db_size]
        sequent_data = self.data_set[initial_db_size:len(self.data_set)]
        tree = Extended_Quant_Tree(self.initial_pi_values)
        tree.build_histogram(initial_data)
        tree.modify_histogram(sequent_data)
        test = Regressed_Change_Detection_Test(tree, len(self.data_set), self.learner)
        value = 0
        for counter in range (number_of_experiments):
            if equal:
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            value += test.reject_null_hypothesis(batch, self.alpha, self.statistic)
        print(value)
        return float(value/number_of_experiments)


def create_bins_combination(bins_number):
    gauss = ccm.random_gaussian(bins_number)
    histogram = np.random.multivariate_normal(gauss[0], gauss[1], 1)
    histogram = histogram[0]
    histogram = np.abs(histogram)
    summa = np.sum(histogram)
    histogram = histogram / summa
    return histogram
