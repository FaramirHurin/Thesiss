import numpy as np
import qtLibrary.libquanttree as qt
from sklearn.neural_network import MLPRegressor
import pickle
from  sklearn.preprocessing import Normalizer
import logging
logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

USES_PICKLE = True

# from main_code import neuralNetworks as nn


# Uses cuts on space instead of the ones on probabilites, equal to normal cut with N=Inf

# Extends canonic quantTree with the possibility to modify the histogram associated
class Incremental_Quant_Tree(qt.QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)
        self.ndata = 0

    def build_histogram(self, data, do_PCA=False):
        super().build_histogram(data, do_PCA)
        self.ndata = len(data)

    def modify_histogram(self, data, definitive=True):
        self.pi_values = self.pi_values * self.ndata
        bins = self.find_bin(data)
        vect_to_add = np.zeros(len(self.pi_values))
        for index in range(len(self.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        self.pi_values = self.pi_values + vect_to_add
        self.ndata = self.ndata + len(data)
        self.pi_values = self.pi_values / self.ndata
        return


class Online_Incremental_QuantTree:
    def __init__(self, pi_values, alpha, statistic):
        self.tree = Incremental_Quant_Tree(pi_values)
        # bins_number = len(pi_values)
        self.alpha = alpha
        self.statistic = statistic
        self.network = Neural_Network()

        # self.network = nn.NN_man(bins_number, 200 * bins_number, 3 * bins_number, 30)
        # self.network.train(alpha)

        self.buffer = None
        self.change_round = None
        self.round = 0
        return

    def build_histogram(self, data):
        self.tree.build_histogram(data)

    def play_round(self, batch):
        threshold = self.network.predict_value(self.tree.pi_values, self.tree.ndata)
        # threshold2 = qt.ChangeDetectionTest(self.tree, len(batch), self.STATISTIC).estimate_quanttree_threshold(self.alpha, 4000)
        stat = self.statistic(self.tree, batch)
        change = stat > threshold
        if not change:
            self.update_model(batch)
        else:
            self.change_round = self.round
        self.round += 1
        return change

    def restart(self):
        self.round = 0
        self.change_round = None
        self.buffer = None

    def update_model(self, batch):
        if self.buffer is not None:
            self.tree.modify_histogram(self.buffer)
        self.buffer = batch
        return


class Neural_Network:

    def __init__(self):
        # sys.path.append('new_testsAndAlgorithms')
        DEBUG = 0
        file = open("dictionary_to_learn.pickle", 'rb')
        self.dictionary = pickle.load(file)
        file.close() #adam, lbfgs, sgd
        self.normal_network = MLPRegressor\
            (hidden_layer_sizes=300, solver='adam', max_iter = 5000, verbose=False, learning_rate_init=0.001, early_stopping=False,
             learning_rate='invscaling', shuffle=False, validation_fraction=0.2, alpha= 0.001, n_iter_no_change=100, random_state=False) #lbfgs
        self.asymptotic_network = MLPRegressor (hidden_layer_sizes=300, solver='adam', max_iter = 5000, verbose=False, learning_rate_init=0.001, early_stopping=True,
             learning_rate='invscaling', shuffle=False, validation_fraction=0.2, alpha= 0.0001, n_iter_no_change=100)
        # logger.debug('Number of nodes is:' + str(self.normal_network.hidden_layer_sizes) + ' and solver is ' + str(self.normal_network.solver) )
        # if self.normal_network.solver != 'lbfgs':
        #     logger.debug('Shuffle is ' + str(self.normal_network.shuffle))
        self.train_network(dictionary=self.dictionary)
        ndata_series = self.dictionary['ndata series']
        self.max_N = max(ndata_series)
        return

    def get_dictionary(self):
        return self.dictionary

    def train_network_normally(self, dictionary):
        normal_bins_series = dictionary['Normal Bins Series']
        normal_thresholds_series = dictionary['Normal thresholds series']
        ndata_series = dictionary['ndata series']
        asymptotic_bins_series = dictionary['Asymptotic bins series']
        asymptotic_thresholds_series = dictionary['Asymptotic thresholds series']
        fat_ndata_Series = (30 * ndata_series)
        flat_ndata_Series = [elem for elem in fat_ndata_Series]
        normal_X_train = [normal_bins_series[index] + (flat_ndata_Series[index],)
                          for index in range(len(normal_bins_series))]

        self.train_normal_network(normal_X_train, normal_thresholds_series)
        self.train_asymptotic_network(asymptotic_bins_series, asymptotic_thresholds_series)
        return

    def train_network(self, dictionary):
        if USES_PICKLE:
            try:
                self.use_stored_network()
            except:
                self.store_trained_network(dictionary)
                self.use_stored_network()
        else:
            self.train_network_normally(dictionary)
        return

    def predict_value(self, bins: np.array, ndata: int):
        if ndata < self.max_N: #or True
            bins = list(bins)
            rich_histogram = bins
            rich_histogram.append(ndata)
            input_data = np.array(rich_histogram)
            # input_data = self.normal_scaler.transform(np.array(rich_histogram).reshape(1, -1))
            return self.normal_network.predict(input_data.reshape(1, -1))
        else:
            # input_data = self.asymptotic_scaler.transform(bins.reshape(1, -1))
            input_data = np.array(bins)
            return self.asymptotic_network.predict(input_data.reshape(1, -1))

    def train_normal_network(self, rich_histograms, thresholds):
        # self.normal_scaler.transform(rich_histograms)
        self.normal_network.fit(rich_histograms, thresholds)
        return

    def train_asymptotic_network(self, histograms, thresholds):
        # self.asymptotic_scaler.transform(histograms)
        self.asymptotic_network.fit(histograms, thresholds)
        return

    def store_trained_network(self, dictionary):
        self.train_network_normally(dictionary)
        file = open('Networks.pickle', 'wb')
        pickle.dump([self.normal_network, self.asymptotic_network], file)
        file.close()
        return

    def use_stored_network(self):
        file = open('Networks.pickle', 'rb')
        [normal, asymptotic] = pickle.load(file)
        file.close()
        self.normal_network = normal
        self.asymptotic_network = asymptotic
        return

