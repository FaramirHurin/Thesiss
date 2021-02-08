
#Objects: tree_creator, Thresholds_numerical_procedure, Regressor, Pi_modifier, Tester

#Interfaces: Threshold_handler_interface must be a common interface for both Thresholds_numerical_procedure and Regressor

import numpy as np
import qtLibrary.libquanttree as qt
import qtLibrary.libccm as ccm
from sklearn.neural_network import MLPRegressor
import pandas as pd


#Hyperparameters
datasets_number = 1000
points_per_batch = 200
bins_number = 8
dimensions_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
alpha = [0.05]
target_sKL = 1
trees_for_the_learner = 5000

#Creates both the original dataset and the batches taken from similar distrbutions

class DataSet_Creator:
    gauss0 = 0
    gauss1 = 0

    def __init__(self):
        self.gauss0 = ccm.random_gaussian(dimensions_number)

#I can call it multiple times, it has always the same gauss0
    def createDataSet(self, datasets_number):
        data = np.random.multivariate_normal(self.gauss0[0], self.gauss0[1], datasets_number)
        return data

    def create_similar_batch(self, target_sKL, nbatch):
        rot, shift = ccm.compute_roto_translation(self.gauss0, target_sKL)
        self.gauss1 = ccm.rotate_and_shift_gaussian(self.gauss0, rot, shift)
        batch = np.random.multivariate_normal(self.gauss1[0], self.gauss1[1], nbatch)
        return batch

    #It uses the multivariate gauss to generate trees
    @classmethod
    def generate_Random_histograms(cls):
        gauss = ccm.random_gaussian(bins_number)
        histograms = np.random.multivariate_normal(gauss[0], gauss[1], trees_for_the_learner)
        histograms = np.abs(histograms)
        hists = []
        for histogram in histograms:
            summa = np.sum(histogram)
            histogram = histogram/summa
            #histogram2 = histogram[0: -1]
            #histogram[-1] = 1 - np.sum(histogram2)
            value_to_subtract = 0
            for binn in range(len(histogram) - 1):
                value_to_subtract = value_to_subtract + histogram[binn]
            histogram[-1] = 1 - value_to_subtract
            value = np.sum(histogram)
            """if not value == 1:
                raise ('Sum exception')
            """
            hists.append(histogram)

        hists = np.array(hists)
        return hists

    @classmethod
    def load_data_set(cls):
        return pd.read_csv('DataBase2')


#Implements algorithm 1 from paper. Given the hyperparameters received with method __init__
#and the dataset we want to build an tree on, it returns the tree created with
#that algorithm, in the form of a QUantTree
class treeCreator:

    binsNumber = 0
    tree = None

    def __init__(self, binsNumber, leaves_probabilites):
        self.binsNumber = binsNumber
        self.quant_tree = qt.QuantTree(leaves_probabilites)
        return

    def createHistogram(self, dataSet):
        self.quant_tree.build_histogram(dataSet)
        return self.quant_tree

#Interface implemented by both Thresholds_numerical_procedure and Regressor.
# Both classes must be able to initialize the hyperparameters, compute the
#threshold for a given statistic and return it
class Threshold_handler_interface:

    def __init__(self):
        pass

    def computeThreshold(self, histogram, ndata):
        pass


#Implements algorithm 2 from the paper in QuantTree
class Thresholds_numerical_procedure(Threshold_handler_interface):

    threshold = 0

    #Set hyperparameters
    def __init__(self, alpha, statistic_used, bins_number):
        self.bins_number = bins_number
        self.alpha = alpha
        self.statistic_used = statistic_used

    #Algorithm 2
    def computeThreshold(self, histogram, ndata):
        tree = qt.QuantTree(histogram)
        tree.ndata = ndata #TODO Rendere sensata questa parte
        if self.statistic_used == 'pearson':
            test = qt.ChangeDetectionTest(tree, 50, qt.pearson_statistic)
        elif self.statistic_used == 'tv':
            test = qt.ChangeDetectionTest(tree, 50, qt.tv_statistic)
        self.threshold = test.estimate_quanttree_threshold(alpha)
        return self.threshold


# The regressor which tries to associate a threshold to a given distribution of the probabilities on
#a tree.
class Regressor(Threshold_handler_interface):

    model = None
    statistic_used = None

    def __init__(self, statistic_used, ndata):
        #TODO Set hyperparameters
        self.model = MLPRegressor(8, solver ='lbfgs', learning_rate = 'adaptive', max_iter = 8100)
        self.statistic_used = statistic_used
        self.tune_Model(ndata)
        return

    def computeThreshold(self, histogram, ndata):
        histogram.sort()
        threshold = self.model.predict(histogram.reshape(1, -1))
        return threshold

    def tune_Model(self, ndata):
        histograms = DataSet_Creator.generate_Random_histograms()
        thresholds = []
        used_histograms = []
        used_thresholds = []
        for histogram in histograms:
            histogram.sort()
            numerical_procedure = Thresholds_numerical_procedure(
                alpha, self.statistic_used, bins_number)
            thresh = numerical_procedure.computeThreshold(histogram, ndata)
            thresholds.append(thresh)
        thresholds = np.array(thresholds)
        histograms = np.array(histograms)
        for index in range(thresholds.size):
            if thresholds[index] < np.percentile(thresholds, 97):
                used_histograms.append(histograms[index])
                used_thresholds.append(thresholds[index])


        self.model.fit(used_histograms, used_thresholds)
        return


#Given a tree created with QuantTree and a dataset of one or multiple data, it must
#update the tree and keep track of the number of data observed so far
# (necessary for updating operations)
class PiModifier():

    tree = None
    number_Of_Data = 0

    def __init__(self, tree, number_Of_Data):
        self.tree = tree
        self.number_Of_Data = number_Of_Data
        return

    #Modifies probabilities
    def modify_probabilities(self, newData):
        self.tree.pi_values = self.tree.pi_values * self.number_Of_Data
        bins = self.tree.find_bin(newData)
        vect_to_add = np.zeros(bins_number)
        for index in range(len(self.tree.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        self.tree.pi_values = self.tree.pi_values +  vect_to_add
        self.number_Of_Data = self.number_Of_Data + len(newData)
        self.tree.pi_values = self.tree.pi_values / self.number_Of_Data
        return self.tree



        #self.tree = (moltiplico per Number of Points i valori di tutti i bins,
        #metto i punti nei bins, divido per il totale e ho le probabilità nuove

#Given the hyperparameters (which it receives with method init) it computes
# the value of the test statistic between a tree and a dataset and tells
# weather the two belong to the same distributions or not.
#It makes use of the threshold computed by the classes implementing Threshold_handler_interface
class Tester:

    statistic_Used = None
    tree = None
    statistic_value = None
    threshold = None

    def __init__(self, statistic_Used, tree, threshold):
        self.statistic_Used = statistic_Used
        self.tree = tree
        self.threshold = threshold
        return

    def get_stat_value(self, dataset):
        if  self.statistic_Used == 'pearson':
            self.statistic_value = qt.pearson_statistic(self.tree, dataset)
        elif self.statistic_Used == 'tv':
            self.statistic_value = qt.tv_statistic(self.tree, dataset)
        return self.statistic_value

    def take_Decision(self, dataset):
        if self.get_stat_value(dataset) < self.threshold:
            return True
        else:
            return False




