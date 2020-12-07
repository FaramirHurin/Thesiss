import numpy as np
import libquanttree as qt
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.neural_network as nn
import sklearn.linear_model as lin
import libccm as ccm
from sklearn import preprocessing
from sklearn import neighbors


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

    def __init__(self, bins_number, statistic, maxN):
        layers_size = bins_number
        self.bins_number = bins_number
        self.statistic = statistic
        self.net = nn.MLPRegressor(layers_size + 1)
        self.asymptpotic_net = nn.MLPRegressor(layers_size)
        self.max_N = maxN
        return

    def tune_model(self, histograms_with_N, thresholds):
        self.net.fit(histograms_with_N, thresholds)
        return

    def tune_model_for_asymptotical_values(self, histograms, thresholds):
        self.asymptpotic_net.fit(histograms, thresholds)
        return

    def predict_threshold(self, histogram, N):
        hist = np.array(list(histogram).append(N))
        if N < self.max_N:
           return self.net.predict(hist.reshape(1, -1))
        else:
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
        if equal:
            return (number_of_experiments - value) / number_of_experiments
        else:
            return value/number_of_experiments

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
        if equal:
            return (number_of_experiments - value) / number_of_experiments, (
                        number_of_experiments - value2) / number_of_experiments
        else:
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
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        if equal:
            return (number_of_experiments - value) / number_of_experiments
        else:
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
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        if equal:
            return (number_of_experiments - value) / number_of_experiments
        else:
            return value / number_of_experiments

    #TODO move here the SKL part
    def run_modified_algorithm_with_learner(self, number_of_experiments, min_N, equal = True):
        initial_db_size = int(len(self.data_set) * self.percentage)
        initial_data = self.data_set[0:initial_db_size]
        sequent_data = self.data_set[initial_db_size:len(self.data_set)]
        tree = Extended_Quant_Tree(self.initial_pi_values)
        tree.build_histogram(initial_data)
        tree.modify_histogram(sequent_data)
        man = NN_man(self.bins_number, self.max_N, 30, 150)
        man.train()
        thr = man.predict_value(tree.pi_values, tree.ndata)
        value = 0
        for counter in range(number_of_experiments):
            if equal:
                batch = self.handler.generate_data_set(self.nu)
            else:
                batch = self.handler.generate_similar_batch(self.nu, self.SKL)
            y = self.statistic(tree, batch)
            value += y > thr
        if equal:
            return (number_of_experiments - value) / number_of_experiments
        else:
            return value / number_of_experiments

def create_bins_combination(bins_number, minN):
    gauss = ccm.random_gaussian(bins_number)
    histogram = np.random.multivariate_normal(gauss[0], gauss[1], 1)
    histogram = histogram[0]
    histogram = np.abs(histogram)
    summa = np.sum(histogram)
    histogram = histogram / summa
    if minN == 0:
        return histogram
    if min(histogram) < 1/minN :
        return create_bins_combination(bins_number, minN)
    histogram = np.sort(histogram)
    return histogram

class NN_man:
    def __init__(self, bins_number, max_N, min_N, nodes):
        self.bins_number = bins_number
        self.standard_learner = nn.MLPRegressor(nodes, solver = 'adam', learning_rate='adaptive', verbose = False, n_iter_no_change=100, max_iter=2000, early_stopping=True)
        self.asymptotic_learner = nn.MLPRegressor(nodes, solver = 'adam', learning_rate='adaptive', verbose=False, n_iter_no_change=100, max_iter=2000, early_stopping=True)
        #self.asymptotic_learner = lin.LinearRegression()
        #self.asymptotic_learner = neighbors.KNeighborsRegressor(2 * nodes, weights='distance')
        self.max_N = max_N
        self.min_N = min_N
        self.asymptotic_normalizer = None
        return

    def compute_dummy_prediction(self):
        histograms, thresholds = self.retrieve_asymptotic_dataSet()
        return float(np.average(thresholds))

    def store_normal_dataSet(self, data_number, nu, statistic, alpha, B):
        histograms_N_thr = np.zeros([data_number, self.bins_number + 2])
        for counter in range(data_number):
            if counter % 10 == 0:
                print (counter)
            N = np.random.randint(self.min_N, self.max_N)
            histogram = create_bins_combination(self.bins_number, self.min_N)
            hist = list(histogram)
            tree = qt.QuantTree(histogram)
            tree.ndata = N
            com = qt.ChangeDetectionTest(tree, nu, statistic)
            threshold = com.estimate_quanttree_threshold(alpha, B)
            hist.append(threshold)
            hist.append(N)
            rich_histogram = np.array(hist)
            rich_histogram = np.sort(rich_histogram)
            histograms_N_thr[counter] = rich_histogram
        frame = pd.DataFrame(histograms_N_thr)
        frame.to_csv('File ending with N and thr')
        return

    def store_asymptotic_dataSet(self, data_number, nu, statistic, alpha, B):
        histograms_thr = np.zeros([data_number, self.bins_number + 1])
        for counter in range(data_number):
            if counter % 1 == 0:
                print (counter)
            histogram = create_bins_combination(self.bins_number, 0)
            alt = Alternative_threshold_computation(histogram, nu, statistic)
            threshold = alt.compute_threshold(alpha, B)
            hist = list(histogram)
            hist.append(threshold)
            histogram = np.array(hist)
            histograms_thr[counter] = histogram
        frame = pd.DataFrame(histograms_thr)
        frame.to_csv('Asymptotic thresholds')
        return

    def retrieve_normal_dataSet(self):
        df = pd.read_csv('File ending with N and thr')
        df_numpy = df.to_numpy()
        thresholds = df_numpy[:, -2]
        histograms = np.delete(df_numpy, -2, 1)
        histograms = np.delete(histograms, 0, 1 )
        return histograms, thresholds

    def retrieve_asymptotic_dataSet(self):
        df = pd.read_csv('Asymptotic thresholds')
        df_numpy = df.to_numpy()
        thresholds = df_numpy[:,-1]
        histograms = np.delete(df_numpy, -1, 1)
        histograms = np.delete(histograms, 0, 1)
        return histograms, thresholds

    def normal_train(self):
        histograms_with_N, thresholds = self.retrieve_normal_dataSet()
        histograms_with_N = np.sort(histograms_with_N)
        print(histograms_with_N[0])
        self.standard_learner.fit(histograms_with_N, thresholds)
        return

    def asymptotic_train(self):
        histograms, thresholds = self.retrieve_asymptotic_dataSet()
        var_min_max = []
        for histogram in histograms:
            variance = np.var(histogram)
            min = np.min(histogram)
            max = np.max(histogram)
            var_min_max.append([variance, min, max])
        var_min_max = np.array(var_min_max)
        histograms = np.sort(histograms)
        self.asymptotic_normalizer = preprocessing.StandardScaler()
        #self.asymptotic_normalizer.fit_transform(histograms)
        self.asymptotic_learner.fit(histograms, thresholds)
        return

    def train(self):
        self.normal_train()
        self.asymptotic_train()


    #Teoria, il rumore interno al singolo istogramma è superiore alla differenza tra istogrammi
    def predict_value(self, histogram, N):
        if N > self.max_N:
            return self.asymptotic_learner.predict(histogram.reshape(1, -1))
        else:
            hist = list(histogram)
            hist.append(N)
            histogram = np.array(hist)
            histogram = np.sort(histogram)
            return self.standard_learner.predict(histogram.reshape(1, -1))
