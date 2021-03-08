import numpy as np
import pandas as pd
from sklearn import neural_network as nn
import qtLibrary.libquanttree as qt
from main_code.auxiliary_project_functions import create_bins_combination, Alternative_threshold_computation
import pickle as pk
import path

PICKLE = True

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
            alternative_computation = Alternative_threshold_computation\
                    (self.histograms[histogram_number], nu, stat)
            threshold = alternative_computation.compute_threshold(alpha, B)
            self.asymptotyical_thresholds[histogram_number] = threshold
        return self.asymptotyical_thresholds

#Provides a good interface to use of the classic
# neural network and train it on the datasets
class NN_man:

    def __init__(self, bins_number, max_N, min_N, nodes):
        self.bins_number = bins_number
        if not PICKLE:
            self.standard_learner = nn.MLPRegressor(nodes, early_stopping=True,
                                                    learning_rate='invscaling', solver='adam',
                                                    validation_fraction=0.1, verbose=False, alpha=0.15,
                                                    max_iter=5000, n_iter_no_change=60)

            self.asymptotic_learner = nn.MLPRegressor(nodes, solver='adam', learning_rate='adaptive',
                                                      verbose=False, n_iter_no_change=60, max_iter=4000,
                                                      early_stopping=False)
        #self.asymptotic_learner = lin.LinearRegression()
        #self.asymptotic_learner = neighbors.KNeighborsRegressor(2 * nodes, weights='distance')
        self.max_N = max_N
        self.min_N = min_N
        self.asymptotic_normalizer = None
        return

    def pickle_learners(self, alpha):
        with open(r'C:\Users\dalun\PycharmProjects\Thesiss\learner_dataset\network.pickle', 'rb') as fil:
            dictionary = pk.load(fil)
            if alpha == [0.01] or alpha == 0.01:
                self.standard_learner = dictionary[('normal', 0.01)]
                self.asymptotic_learner = dictionary[('asymptotic', 0.01)]
            else:
                self.asymptotic_learner = dictionary[('Asymptotic', 0.5)]
                self.standard_learner = dictionary[('normal', 0.5)]
            # print('I pickle')
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
        alpha = alpha[0]
        if alpha == 0.5:
            frame.to_csv(r'C:\Users\dalun\PycharmProjects\Thesiss\File_N_and_thr_0_5_pearson')
        elif alpha == 0.01:
            frame.to_csv(r'C:\Users\dalun\PycharmProjects\Thesiss\File_N_and_thr_0_01_pearson')
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
        alpha = alpha[0]
        if alpha == 0.5:
            frame.to_csv(r'C:\Users\dalun\PycharmProjects\Thesiss\Asymptotic_0_5')
        elif alpha == 0.01:
            frame.to_csv(r'C:\Users\dalun\PycharmProjects\Thesiss\Asymptotic_0._1')
        return

    def retrieve_normal_dataSet(self, alpha):
        if not isinstance(alpha, float):
            alpha = alpha[0]
        #df = pd.read_csv('File ending with N and thr')
        if alpha == 0.01:
            df = pd.read_csv('..\learner_dataset\File_N_and_thr_0_01')
        elif alpha == 0.5:
            df = pd.read_csv('..\learner_dataset\File_N_and_thr_0_5')
        else:
            raise Exception('alpha is wrong, expected 0.01 or 0.5, got' + str(alpha))
        df_numpy = df.to_numpy()
        thresholds = df_numpy[:, -2]
        histograms = np.delete(df_numpy, -2, 1)
        histograms = np.delete(histograms, 0, 1 )
        return histograms, thresholds

    def retrieve_asymptotic_dataSet(self, alpha):
        if not isinstance(alpha, float):
            alpha = alpha[0]
        if alpha == 0.01:
            df = pd.read_csv(r'../learner_dataset/Asymptotic_0._1')
        elif alpha == 0.5:
            df = pd.read_csv(r'../learner_dataset/Asymptotic_0_5')
        else:
            raise Exception('alpha is wrong, expected 0.1 or 0.5, got' + str(alpha))
        df_numpy = df.to_numpy()
        thresholds = df_numpy[:,-1]
        histograms = np.delete(df_numpy, -1, 1)
        histograms = np.delete(histograms, 0, 1)
        return histograms, thresholds

    def normal_train(self, alpha):
        histograms_with_N, thresholds = self.retrieve_normal_dataSet(alpha)
        histograms_with_N = np.sort(histograms_with_N)
        self.standard_learner.fit(histograms_with_N, thresholds)
        return

    def asymptotic_train(self, alpha):
        histograms, thresholds = self.retrieve_asymptotic_dataSet(alpha)
        histograms = np.sort(histograms)
        #self.asymptotic_normalizer = preprocessing.StandardScaler()
        #self.asymptotic_normalizer.fit_transform(histograms)
        self.asymptotic_learner.fit(histograms, thresholds)
        return

    def train(self, alpha):
        if PICKLE:
            self.pickle_learners(alpha)
            return
        self.normal_train(alpha)
        self.asymptotic_train(alpha)
        return

    #Teoria, il rumore interno al singolo istogramma è superiore alla differenza tra istogrammi
    def predict_value(self, histogram, N):

        if N > self.max_N: # and False:
            return self.asymptotic_learner.predict(histogram.reshape(1, -1))

        else:
            hist = list(histogram)
            hist.append(N)
            histogram = np.array(hist)
            histogram = np.sort(histogram)
            return self.standard_learner.predict(histogram.reshape(1, -1))


