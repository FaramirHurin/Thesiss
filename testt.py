import auxilium as aux
import auxilium_test as aux_test
import libquanttree as qt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

percentage = 0.01
bins_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 1000
alpha = [0.005]
data_Dimension = 2
nu = 32
B = 3000
statistic = qt.tv_statistic
X = [3]
data_number_for_learner = 1000
max_N = 2000
min_N = 50
SKL = 1

# Prove 1: Variance decreases when B increases
#Prove 2: For alternative(and normal) the exixtence of some crazy high value is sistematic

def test_asymptotic_coherence():
    means = []
    variances = []
    for counter in range(10):
        bins = aux.create_bins_combination(aux_test.bins_number, 100)
        alt = aux.Alternative_threshold_computation(bins, nu, statistic)
        thresholds = []
        for inner_counter in range(10):
            threshold = alt.compute_threshold(alpha, B)
            thresholds.append(threshold)
        means.append(np.mean(thresholds))
        variances.append(np.var(thresholds))
    print('Means '+ str(means))
    print ('Variances' + str(variances))

def test_standard_coherence(min_N, max_N):
    means = []
    variances = []
    for count in range(10):
        N = np.random.randint(min_N, max_N)
        tree = qt.QuantTreeUnivariate(aux.create_bins_combination(bins_number, min_N))
        tree.ndata = N
        thresholds = []
        for innercount in range(10):
            test = qt.ChangeDetectionTest(tree, nu, statistic)
            threshold = test.estimate_quanttree_threshold(alpha, B)
            thresholds.append(threshold)
        means.append(np.mean(thresholds))
        variances.append(np.var(thresholds))
    print ('Means are' + str(means))
    print ('Variances are' + str(variances))
    return

#Observe variance decrreasing when B is increasing.
#Probably given the current setting, B = 3000 is enough to almost plateau the noise problem
def observe_growing_B(bins, B_values, N, experiments_number):
    variances = []
    means = []
    for B_value in B_values:
        tree = qt.QuantTreeUnivariate(bins)
        tree.ndata = N
        test = qt.ChangeDetectionTest(tree, nu, statistic)
        thresholds = []
        for counter in range(experiments_number):
            thresholds.append(test.estimate_quanttree_threshold(alpha, B_value))
        variances.append(np.var(thresholds))
        means.append(np.mean(thresholds))
    plt.plot(B_values, variances)
    plt.title('Variance on growing B')
    plt.plot(B_values, means)
    plt.title('Means on growing B ')
    plt.show()
    return

#For asymptotic computation probably B = 5000 is enough
def observe_growing_asymptpotic_B(bins, B_values, N, experiments_number):
    variances = []
    means = []
    for B_value in B_values:
        tree = qt.QuantTreeUnivariate(bins)
        alt = aux.Alternative_threshold_computation(bins, nu, statistic)
        thresholds = []
        for counter in range(experiments_number):
            thresholds.append(alt.compute_threshold(alpha,B_value))
        variances.append(np.var(thresholds))
        means.append(np.mean(thresholds))
    plt.plot(B_values, variances)
    plt.title('Asymptotic ariance on growing B')
    plt.plot(B_values, means)
    plt.title('Means on growing B ')
    plt.show()
    return

#We proved that the crazy high values are sistematic
def try_asymptotic_correctness():
    man = aux.NN_man(bins_number, 100, 10, 20)
    histograms, thresholds = man.retrieve_asymptotic_dataSet()
    for count in range(len(thresholds)):
        threshold = thresholds[count]
        if threshold > np.quantile(thresholds, 0.99):
            alt = aux.Alternative_threshold_computation(histograms[count], nu, statistic)
            thr = alt.compute_threshold(alpha, 3000)
            print ('Normal value and predicted value')
            print(threshold, thr)

def test_histogram_with_bins(bins, truth):
    superman = aux.Superman(percentage=0.01,  SKL = 4, initial_pi_values = bins, data_number = 5000,
                 alpha = alpha, bins_number = bins_number, data_Dimension = 3, nu = nu, B = B, statistic = statistic, max_N = 1000,
                 data_number_for_learner = 100)
    superman.create_training_set_for_QT()
    superman2 = aux.Superman(percentage=0.01,  SKL = 4, initial_pi_values = aux_test.initial_pi_values, data_number = 5000,
                 alpha = alpha, bins_number = bins_number, data_Dimension = 3, nu = nu, B = B, statistic = statistic, max_N = 1000,
                 data_number_for_learner = 100)
    superman2.create_training_set_for_QT()
    return superman.run_normal_algorithm(5000, truth), superman2.run_normal_algorithm(5000, truth)

#Confirmed: histograms correlated with high (asymptotic) thresholds makes the model have a bad performance
def control_high_thresholds():
    print('Ciao')
    man = aux.NN_man(bins_number, 100, 10, 20)
    histograms, thresholds = man.retrieve_asymptotic_dataSet()
    for count in range(len(thresholds)):
        threshold = thresholds[count]
        if threshold > np.quantile(thresholds, 0.99):
            FP0s = test_histogram_with_bins(histograms[count], True)
            powers = test_histogram_with_bins(histograms[count], False)
            print ('Threshold is' + str(threshold))
            print('powers are ' + str(powers))
            print('FP0s are ' + str(FP0s))

def try_normal_correctness():
    man = aux.NN_man(bins_number, 100, 10, 20)
    histograms, thresholds = man.retrieve_asymptotic_dataSet()
    histograms = np.delete(histograms, -1, 1)
    for count in range(len(thresholds)):
        threshold = thresholds[count]
        if threshold > np.quantile(thresholds, 0.99):
            alt = aux.Alternative_threshold_computation(histograms[count], nu, statistic)
            thr = alt.compute_threshold(alpha, 1000)
            print ('Normal value and predicted value')
            print(threshold, thr)
    return

#Sembrerebbe esserci una correlazione molto forte tra gli outliers e il minimo bin molto piccolo
def test_high_thresholds_histograms():
    print('Ciao')
    man = aux.NN_man(bins_number, 100, 10, 20)
    histograms, thresholds = man.retrieve_asymptotic_dataSet()
    mins = []
    variances = []
    for histogram in histograms:
        mins.append(np.min(histogram))
        variances.append(np.var(histogram))
    for count in range(len(thresholds)):
        threshold = thresholds[count]
        if threshold > np.quantile(thresholds, 0.995):
            print('Mins')
            print(np.mean(mins), np.min(histograms[count]))
            print('Variances')
            print (np.mean(variances), np.var(histograms[count]))

#Tests the R2 of the predictions made by the NN
def test_asymptotic_training(number_of_experiments):
    man = aux.NN_man(bins_number, 20, 1000, 1000)
    man.asymptotic_train()
    thresholds = []
    predictions = []
    for counter in range(number_of_experiments):
        histogram = aux.create_bins_combination(bins_number, 20)
        thresholds.append(aux.Alternative_threshold_computation(histogram, nu, statistic).compute_threshold(alpha, B))
        predictions.append(float(man.predict_value(histogram, 2000)))
    print ('R2 is')
    print(r2_score(thresholds, predictions))
    print ('Thresholds')
    print(thresholds)
    print('Predictions')
    print(predictions)

def test_normal_training(number_of_experiments, minN, maxN):
    man = aux.NN_man(bins_number, 1000, 50, 120)
    man.normal_train()
    thresholds = []
    predictions = []
    for counter in range(number_of_experiments):
        histogram = aux.create_bins_combination(bins_number, 20)
        N = np.random.randint(minN, maxN)
        tree = qt.QuantTreeUnivariate(histogram)
        tree.ndata = N
        test = qt.ChangeDetectionTest(tree, nu, statistic)
        thresholds.append(test.estimate_quanttree_threshold(alpha, B))
        predictions.append(float(man.predict_value(histogram, N)))
    print('R2 is')
    print(r2_score(thresholds, predictions))
    print('Thresholds')
    print(thresholds)
    print('Predictions')
    print(predictions)



def compare_FP0s(algorithms, number_of_experiments, batches_per_experiment):
    superman = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                 alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                 data_number_for_learner)
    results = np.zeros([number_of_experiments, len(algorithms)])
    for counter in range(number_of_experiments):
        superman.create_training_set_for_QT()
        for algorithm_number in range(len(algorithms)):
            if algorithms[algorithm_number] == 'normal':
                results[counter][algorithm_number] = \
                    superman.run_modified_algorithm_without_learner(batches_per_experiment, min_N)[1]
            elif algorithms[algorithm_number] == 'modified':
                results[counter][algorithm_number] = \
                    superman.run_modified_algorithm_with_learner(batches_per_experiment, min_N)[0]
            elif algorithms[algorithm_number] == 'asymptotic':
                results[counter][algorithm_number] = \
                    superman.run_asymtpotic_algorithm_without_learner(batches_per_experiment)
            elif algorithms[algorithm_number] == 'regressor':
                results[counter][algorithm_number] = \
                    superman.run_modified_algorithm_with_learner(batches_per_experiment, aux_test.min_N)
    plt.fig, axs = plt.subplots(1, len(algorithms))
    results = np.transpose(results)
    for alg in range(len(algorithms)):
        axs[alg].boxplot(results[alg])
        axs[alg].set_title('FP0 '+ str(algorithms[alg]))
    plt.show()
    return


def compare_powers(algorithms, number_of_experiments, batches_per_experiment, SKL):
    superman = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                 alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                 data_number_for_learner)
    results = np.zeros([number_of_experiments, len(algorithms)])
    for counter in range(number_of_experiments):
        superman.create_training_set_for_QT()
        for algorithm_number in range(len(algorithms)):
            if algorithms[algorithm_number] == 'normal':
                results[counter][algorithm_number] = \
                    superman.run_modified_algorithm_without_learner(batches_per_experiment, False)[1]
            elif algorithms[algorithm_number] == 'modified':
                results[counter][algorithm_number] = \
                    superman.run_modified_algorithm_without_learner(batches_per_experiment, False)[0]
            elif algorithms[algorithm_number] == 'asymptotic':
                results[counter][algorithm_number] = \
                    superman.run_asymtpotic_algorithm_without_learner(batches_per_experiment, False)
            elif algorithms[algorithm_number] == 'regressor':
                results[counter][algorithm_number] = \
                    superman.run_modified_algorithm_with_learner(batches_per_experiment, aux_test.min_N, False)
    plt.fig, axs = plt.subplots(1, len(algorithms))
    for alg in range(len(algorithms)):
        axs[alg].boxplot(results[:, alg])
        axs[alg].set_title('Power'+ str(algorithms[alg]))
    plt.show()
    return



"""
bins = aux_test.initial_pi_values
observe_growing_asymptpotic_B(bins,[2000, 3000, 4000, 5000, 6000], 500, 100)
"""
#control_high_thresholds()
#test_high_thresholds_histograms()
#try_normal_correctness()
#test_normal_training(100, 50, 1000)
#test_asymptotic_training(1000)\\

print('Ciao')
compare_FP0s(['normal', 'modified', 'asymptotic', 'regressor'], 300, 3000)
compare_powers(['normal', 'modified', 'asymptotic', 'regressor'], 300, 3000, 2)
