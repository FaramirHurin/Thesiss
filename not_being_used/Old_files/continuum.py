

from old_files.scratch import *
import numpy as np
import qtLibrary.libquanttree as qt
import pandas as pd
import matplotlib.pyplot as plt

#Hyperparams
#X = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
X = [5, 6, 7, 8, 9, 10]
N = np.power(2, X)
B = 1000
#pi_values_vector = DataSet_Creator.generate_Random_histograms()
nu = 100

# v = 64 ?????

#Compute thresholds for every histogram and N?
#Do threshold's variance reduce when moving towards biggern N?
def create_thresholds_on_growing_N(alpha, N, B):
    pi_values_vector = DataSet_Creator.generate_Random_histograms()
    thresholds = np.empty((len(N), len(pi_values_vector)))
    number_counter = 0
    for number in N:
        print (number)
        pi_counter = 0
        for pi_values in pi_values_vector:
            #print('Run number ' + str(pi_counter))
            pi_values = np.array(pi_values)
            tree = qt.QuantTree(pi_values)
            tree.ndata = number
            test = qt.ChangeDetectionTest(tree, nu, qt.tv_statistic)
            threshold = test.estimate_quanttree_threshold(alpha, B)
            thresholds[number_counter][pi_counter] = threshold
            pi_counter = pi_counter + 1
        number_counter = number_counter + 1
    frame = pd.DataFrame(thresholds)
    frame.to_csv('Thresholds_for_different_N_3')

    #If N>N' -> var(thresholds) < var(thresholds') return true
    return False


def plot_on_growing_N():
    df = pd.read_csv('Thresholds_for_different_N_3')
    df = df.to_numpy()
    to_plot = np.transpose(df)
    plt.boxplot(to_plot, notch=True)
    plt.show()

def test_FPO_for_N():
    ndata = 300
    numerical_procedure = Thresholds_numerical_procedure(alpha, 'pearson', bins_number)
    histograms = DataSet_Creator.generate_Random_histograms()
    dataset_creator = DataSet_Creator()
    training_set = dataset_creator.createDataSet(trees_for_the_learner)
    batches = []
    batches_number = 500
    #batches creation
    for count in range(batches_number):
        batches.append(dataset_creator.createDataSet(50))
    #Thresholds computation
    thrsholds = np.ones((len(N),trees_for_the_learner))
    for size in range(len(N)):
        for hist in range(trees_for_the_learner):
            thrsholds[size][hist] = numerical_procedure.computeThreshold(histograms[hist], N[size])
        print(size)
    trees= []
    print('trees creation')
    for hist in range(trees_for_the_learner):
        if hist % 5 == 0:
            print(hist)
        tree_Creator = treeCreator(bins_number, histograms[hist])
        tree = tree_Creator.createHistogram(training_set)
        trees.append(tree)
    recognized_percentages = []
    print ('control')
    for size in range(len(N)):
        print(size)
        recognized_percentage = 0
        for hist in range(trees_for_the_learner):
            if hist%5 == 0:
                print(hist)
            tester = Tester('pearson', trees[hist], thrsholds[size][hist])
            count = 0
            for batch in batches:
                if tester.take_Decision(batch) == True:
                    count = count + 1
            recognized_percentage = count/batches_number
        recognized_percentages.append(recognized_percentage)
    plt.plot(N, recognized_percentages, label ='Percentage recognized using N')
    plt.legend()
    plt.show()

#Second test
# Compute thresholds in a different way, splitting on space and not prob.
#Is threshold's variance smaller than other ways?
def test_variance_on_space_split(pi_values, alpha, v, B):
    #if variance<all variances return True
    return False


#Third test
def definitive_test(pi_values, histograms):
    #for some elem in pivalues+histograms: test_case
    return


#Auxilary test:
#Given pi_values and threshold, test FPO and and power
def test_case(pi_values, threshold):
    #plots performance
    return



#create_thresholds_on_growing_N(alpha, N, B)
plot_on_growing_N()
#test_FPO_for_N()