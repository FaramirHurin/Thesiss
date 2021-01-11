import extendedQuantTree as aux
import dynamicQuantTree
import qtLibrary.libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt

#HYPERPARAMETERS
import neuralNetworks
import superman

percentage = 0.1
bins_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 200
alpha = [0.01]
data_Dimension = 3
nu = 32
B = 3000
statistic = qt.tv_statistic
X = [3]
data_number_for_learner = 10000
max_N = bins_number
min_N = nu
SKL = 1

#WE ARE STORING WITH NU = 32

#Extended QuantTree tests

#Compare FP0 between QT and Extended QT without NN
def compare_FP0(SKL):
    number_of_tests_for_the_plot = 10
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 2000

    test = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        result = test.run_modified_algorithm_without_learner(number_of_batches_per_test)
        modified_value += result[0]
        normal_value += result[1]
        normal_to_plot.append(1 - normal_value)
        modified_to_plot.append(1 - modified_value)
    plt.boxplot([normal_to_plot, modified_to_plot], labels=['normal', 'modified'])
    plt.title('FPO: normal and extended')
    plt.show()
    return

#Compare power between QT and Extended QT without NN
def compare_power(SKL):
    number_of_tests_for_the_plot = 10
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 2000

    test = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        v1, v2 = test.run_modified_algorithm_without_learner\
            (number_of_batches_per_test, False)
        normal_value += v2
        modified_value += v1
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)

    print ('Powers are: normal = ' + str(np.mean(normal_to_plot)) + 'modified is' + str(np.mean(modified_to_plot)))
    plt.boxplot([normal_to_plot, modified_to_plot],
                labels=['normal', 'modified'], notch=False)
    plt.title('Power: normal and modified')
    plt.show()
    return normal_to_plot, modified_to_plot

#Compare FP0 between QT and Extended QT without NN.
#We use the asymptotic threshold computation and many points
def compare_FP0_with_asymptotic(SKL):
    print (str(data_number) + ' dataNumber')
    number_of_tests_for_the_plot = 10
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 10000

    test = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test)
        modified_value += test.run_asymtpotic_algorithm_without_learner(number_of_batches_per_test)
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)
    print ('Given this quantity of data ' + str(data_number) + ' the means for the FP0 are: asymptotic  and normal')
    print (np.mean(modified_to_plot), np.mean(normal_to_plot))
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard FP0')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Extended with asymptotic FPO')
    plt.show()
    return

#Compare power between QT and Extended QT without NN.
#We use the asymptotic threshold computation and many points
def compare_power_with_asymptotic(SKL):
    number_of_tests_for_the_plot = 10
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 2000

    test = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test, False)
        modified_value += test.run_asymtpotic_algorithm_without_learner(number_of_batches_per_test, False)
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)
    print('The means for the powers are: asymptotic and normal')
    print(np.mean(modified_to_plot), np.mean(normal_to_plot))
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard power')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Extended with asymptotic power')
    plt.show()
    return

#Compare FP0 between QT and Extended QT with NN
def compare_regressor_FP0(SKL):
    number_of_tests_for_the_plot = 50
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 3000

    test = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        print('experiment ' + str(index))
        test.create_training_set_for_QT()
        normal_value = test.run_modified_algorithm_without_learner(number_of_batches_per_test)[1]
        modified_value = float(test.run_modified_algorithm_with_learner(number_of_batches_per_test, min_N, alpha))
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)

    print ('Plotting regressor FP0')
    plt.boxplot([normal_to_plot, modified_to_plot], labels= ['normal', 'with regressor'])
    plt.title('Regressor FP0')
    plt.show()
    return

#Compare power between QT and Extended QT without NN
def compare_regressor_power(SKL):
    number_of_tests_for_the_plot = 40
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 1000

    test = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             data_number_for_learner)

    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = test.run_normal_algorithm(number_of_batches_per_test, False)
        modified_value = test.run_modified_algorithm_with_learner(number_of_batches_per_test, 20, alpha, False)
        normal_to_plot.append(normal_value)
        modified_to_plot.append(float(modified_value))

    print ('Power')
    plt.boxplot([normal_to_plot, modified_to_plot], labels=['normal', 'regressed'])
    plt.title('Regressor: power')
    plt.show()

def store_datestets():
    nodes = 3
    n = neuralNetworks.NN_man(bins_number, max_N, min_N, nodes)
    #n.store_normal_dataSet(data_number_for_learner, nu, statistic, [0.01], 4000)
    #n.store_normal_dataSet(data_number_for_learner, nu, statistic, [0.5], 4000)
    n.store_asymptotic_dataSet(int(data_number_for_learner), nu, statistic, [0.01], 5000)
    n.store_asymptotic_dataSet(int(data_number_for_learner), nu, statistic, [0.5], 5000)

#Dynamic QuantTree tests

#Plots the EWMA value of one run.
def test_EWMA_once(run_lenght):
    data = aux.Data_set_Handler(data_Dimension)
    statistics = np.zeros(run_lenght)
    x = dynamicQuantTree.DynamicQuantTree(initial_pi_values, 0.3, qt.tv_statistic, 1000, 100, alpha, False, False)
    for round in range(run_lenght):
        batch = data.return_equal_batch(nu)
        statistics[round] = x.playRound(batch)[0]
    return statistics

#Averages over multiple runs. We expect the EWMA to be around alpha for every time T
def test_EWMA():
    print('ciao')
    experiments_number = 200
    run_lenght =1000
    statistics = np.zeros(run_lenght)
    for exp in range(experiments_number):
        statistics = statistics + test_EWMA_once(run_lenght)
    statistics = statistics/experiments_number
    plt.plot(range(len(statistics)), statistics)
    plt.title('stat value')
    plt.show()

def plot_EWMA_ARL0(max_run_lenght):
    data = aux.Data_set_Handler(data_Dimension)
    statistics = np.zeros(max_run_lenght)
    x = dynamicQuantTree.DynamicQuantTree(initial_pi_values, 0.3, qt.tv_statistic, 1000, 100, alpha, True, False)
    for round in range(max_run_lenght):
        batch = data.return_equal_batch(nu)
        statistics[round], stop = x.playRound(batch)
        if stop:
            break
    return round

def plot_EWMA_ARL1(SKL, max_run_lenght):
    return

test_EWMA()