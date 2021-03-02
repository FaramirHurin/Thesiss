import qtLibrary.libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt
from main_code import neuralNetworks, auxiliary_project_functions, incremental_QuantTree as aux
import not_being_used.Old_files.superman as superman
from main_code.auxiliary_project_functions import create_bins_combination

percentage = 0.9
bins_number = 8
initial_pi_values = create_bins_combination(bins_number)
data_number = 500
alpha = [0.01]
beta = 0.1
data_Dimension = 3
nu = 32
B = 4000
statistic = qt.tv_statistic
X = [3]
data_number_for_learner = 600
max_N = 3000
min_N = nu
SKL = 1

#WE ARE STORING WITH NU = 32

#Extended QuantTree tests

#Compare FP0 between QT and Extended QT without NN
def compare_FP0(SKL):
    number_of_tests_for_the_plot = 10
    normal_to_plot = []
    pearson_to_plot = []
    tv_to_plot = []
    number_of_batches_per_test = 6000

    test_pearson = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                                     alpha, bins_number, data_Dimension, nu, B, qt.pearson_statistic, max_N,
                                     data_number_for_learner)
    test_tv = superman.Superman(percentage, SKL, initial_pi_values, data_number,
                                alpha, bins_number, data_Dimension, nu, B, qt.tv_statistic, max_N,
                                data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test_pearson.create_training_set_for_QT()
        test_tv.create_training_set_for_QT()
        normal_value = 0
        pearson_value = 0
        tv_value = 0
        pearson_result = test_pearson.run_modified_algorithm_without_learner(number_of_batches_per_test)
        tv_result = test_tv.run_modified_algorithm_without_learner(number_of_batches_per_test)
        pearson_value += pearson_result[0]
        normal_value += tv_result[1]
        tv_value += tv_result[0]
        normal_to_plot.append(normal_value)
        pearson_to_plot.append(pearson_value)
        tv_to_plot.append(tv_value)

    plt.boxplot([normal_to_plot, pearson_to_plot, tv_to_plot],
                labels=['normal', 'pearson', 'tv'], showfliers=True, showmeans=True)
    plt.title('FPR')
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
    number_of_tests_for_the_plot = 10
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

    print ('Plotting regressor FPR')
    plt.boxplot([normal_to_plot, modified_to_plot], labels= ['normal', 'with regressor'], showmeans= True)
    plt.title('Normal and regressor FPR: target 0.01')
    plt.show()
    return

#Compare power between QT and Extended QT without NN
def compare_regressor_power(SKL):
    number_of_tests_for_the_plot = 10
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

    return normal_to_plot, modified_to_plot
    """
    print ('Power')
    plt.boxplot([normal_to_plot, modified_to_plot], labels=['normal', 'regressed'])
    plt.title('Regressor: power')
    plt.show()
    """

def store_datestets():
    nodes = 3
    n = neuralNetworks.NN_man(bins_number, max_N, min_N, nodes)
    n.store_normal_dataSet(data_number_for_learner, nu, statistic, [0.01], 4000)
    n.store_normal_dataSet(data_number_for_learner, nu, statistic, [0.5], 4000)
    n.store_asymptotic_dataSet(int(data_number_for_learner), nu, statistic, [0.01], 5000)
    n.store_asymptotic_dataSet(int(data_number_for_learner), nu, statistic, [0.5], 5000)

def single_alternative_FP0_comparison(batches, statistic):
    values = np.zeros(batches)
    pi_values = auxiliary_project_functions.create_bins_combination \
        (len(initial_pi_values), 2 * len(initial_pi_values))
    tree = aux.Incremental_Quant_Tree(initial_pi_values)
    data_generator = auxiliary_project_functions.Data_set_Handler(data_number)
    training_set = data_generator.return_equal_batch(1000)
    tree.build_histogram(training_set)
    threshold = qt.ChangeDetectionTest(tree, nu, statistic).estimate_quanttree_threshold(alpha, B)
    for index in range(batches):
        batch = data_generator.return_equal_batch(nu)
        values[index] = statistic(tree, batch) > threshold
    return values

def alternative_FP0_comparison(batches, points_to_plot):
    box_pearson = np.zeros(points_to_plot)
    box_tv = np.zeros(points_to_plot)
    for index in range(points_to_plot):
        pearson_value = np.mean(single_alternative_FP0_comparison(batches, qt.pearson_statistic))
        print('Pearson: ' + str(pearson_value))

        tv_value = np.mean(single_alternative_FP0_comparison(batches, qt.tv_statistic))
        print('TV: ' + str(tv_value))

        box_pearson[index] = pearson_value
        box_tv[index] = tv_value

    plt.boxplot([box_pearson, box_tv], labels=['Pearson', 'TV'])
    plt.title('False Positive Rate: random pi_values - 8 bins')
    plt.show()

#compare_regressor_FP0(1)

max_SKL = 10
normals = np.zeros(max_SKL-1)
changed = np.zeros(max_SKL-1)
for SKL in range(1, max_SKL):
    norm, mod = compare_regressor_power(SKL)
    normals[SKL-1] = np.average(norm)
    changed[SKL-1] = np.average(mod)
plt.plot(normals, label = 'Classic QT')
plt.plot(changed, label = 'Regressor')
plt.ylabel('Power')
plt.xlabel('Change Magnitude')
plt.title('Normal and regressor powers')
plt.show()