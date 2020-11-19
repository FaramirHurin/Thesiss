import auxilium as aux
import libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt

#HYPERPARAMETERS
percentage = 0.2
SKL = 6
bins_number = 16
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 1000
alpha = [0.05]
data_Dimension = 8
nu = 32
B = 1000
statistic = qt.pearson_statistic
X = [3]
N_values = np.power(2, X)
data_number_for_learner = 500

#Old test, must beeliminated when the new is ready
def testFP0():
    #Hyperparameters
    number_of_training_sets = 10
    number_of_tests_per_training = 10
    dimensions_number = 6
    bins_number = 8
    data_number = 100000
    B = 600
    percentage_immediately_observed = 0.1
    alpha = [0.01]
    statistic =  qt.tv_statistic

    standard_pi_values = np.ones(bins_number)/bins_number
    FP0_values_standard = []
    FP0_values_extended = []
    for experiment in range(number_of_training_sets):
        data_set_handler = aux.Data_set_Handler(dimensions_number)
        training_set = data_set_handler.generate_data_set(data_number)
        partial_training_set = training_set[0: int(percentage_immediately_observed*data_number)]
        rest_of_the_training_set = training_set[int(percentage_immediately_observed*data_number)+1: data_number]

        modified_tree = aux.Extended_Quant_Tree(standard_pi_values)
        modified_tree.build_histogram(partial_training_set)
        modified_tree.modify_histogram(rest_of_the_training_set)
        standard_tree = qt.QuantTree(modified_tree.pi_values)
        standard_tree.build_histogram(training_set)

        acceptance_record_standard = 0
        acceptance_record_modified = 0

        for batch_number in range(number_of_tests_per_training):
            batch = data_set_handler.return_equal_batch(B)
            standard_test = qt.ChangeDetectionTest(standard_tree, B, statistic)
            extended_test = qt.ChangeDetectionTest(modified_tree, B, statistic)
            extended_threshold = extended_test.estimate_quanttree_threshold(alpha)
            standard_threshold = standard_test.estimate_quanttree_threshold(alpha)

            if extended_threshold > statistic(modified_tree, batch):
                acceptance_record_modified += 1
            if standard_threshold > statistic(standard_tree, batch):
                acceptance_record_standard += 1
        experiment_standard_value = acceptance_record_standard/number_of_tests_per_training
        experiment_extended_value = acceptance_record_modified/number_of_tests_per_training

        FP0_values_extended.append(experiment_extended_value)
        FP0_values_standard.append(experiment_standard_value)
    print('Extended')
    print (FP0_values_extended)
    print('Standard')
    print(FP0_values_standard)
    fig, axs = plt.subplots(2)
    axs[0].boxplot(FP0_values_standard, notch = False)
    axs[0].set_title('Standard')
    axs[1].boxplot(FP0_values_extended, notch = False)
    axs[1].set_title('Extended')
    plt.show()

def test_cuts():
    pi_values = aux.create_bins_combination(bins_number)
    pi_values = np.sort(pi_values)
    computation = aux.Alternative_threshold_computation(pi_values, nu, statistic)
    tree = qt.QuantTreeUnivariate(pi_values)
    data = np.random.uniform(0, 1, 100000)
    tree.build_histogram(data)
    cut = computation.compute_cut()

    for index1 in range(len(cut.leaves)-1):
        print(cut.leaves[index1+1]-cut.leaves[index1], tree.leaves[index1+1]-tree.leaves[index1])
    print('Asymptotic leaves')
    print(cut.leaves)
    print('Normal leaves')
    print(tree.leaves)

""" Generate a dataset of points. Train the classic agorithm and the one without the NN.
Test the performance of the two on the same batches and plot the reuslt.
Result: Two box plots
"""
def compare_FP0():
    number_of_tests_for_the_plot = 100
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 2000

    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                 alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
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
    print ('An I plotting?')
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard FP0')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Extended FP0')
    plt.show()
    return

def compare_FP0_with_asymptotic():
    number_of_tests_for_the_plot = 50
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 2000

    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                 alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                 data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test)
        modified_value += test.run_asymtpotic_algorithm_without_learner(number_of_batches_per_test)
        normal_to_plot.append(1 - normal_value)
        modified_to_plot.append(1 - modified_value)
    print ('The means are: modified and normal')
    print (np.mean(modified_to_plot), np.mean(normal_to_plot))
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard FP0')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Extended with asymptotic FPO')
    plt.show()
    return

def compare_power():
    number_of_tests_for_the_plot = 1
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 1000

    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                        alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                        data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test, False)
        modified_value += test.run_modified_algorithm_without_learner(number_of_batches_per_test, False)[0]
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)

    #normal_to_plot = np.ones(len(normal_to_plot)) - normal_to_plot
    #modified_to_plot = np.ones(len(modified_to_plot)) - modified_to_plot

    print ('Am I plotting? 2')
    #fig, axs = plt.subplots(2)
    #axs[0].boxplot(normal_to_plot, notch=False)
    #axs[0].set_title('Standard power')
    #axs[1].boxplot(modified_to_plot, notch=False)
    #axs[1].set_title('Extended power')
    #plt.show()
    return normal_to_plot, modified_to_plot

def compare_regressor_FP0():
    number_of_tests_for_the_plot = 1
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 1000

    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                        alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                        data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        print('experiment ' + str(index))
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test)
        modified_value += test.run_modified_algorithm_with_learner(number_of_batches_per_test)
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)

    print ('Am I plotting 3')
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard FP0')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Regressor FP0')
    plt.show()
    return

def compare_regressor_power():
    number_of_tests_for_the_plot = 1
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 1000

    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                        alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                        data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        test.create_and_train_net()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test, False)
        modified_value += test.run_modified_algorithm_with_learner(number_of_batches_per_test, False)
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)

    print ('Am I plotting 4')
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard: 1 - power')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Regressor: 1 - power')
    plt.show()

"""Generate N1 and N2 data points. For both generate B data points and plot 
the cumulative functuons.
"""
def plot_cumulative_functions(N1, N2, B, pi_values, nu, statistic):
    to_average = 200
    st1 = []
    st2 = []
    def_values1 = np.zeros(B)
    def_values2 = np.zeros(B)
    for indice in range(to_average):
        pi_values = aux.create_bins_combination(bins_number)
        statistics1 = []
        statistics2 = []
        data_set1 = np.random.uniform(0, 1, N1)
        tree1 = qt.QuantTreeUnivariate(pi_values)
        tree1.build_histogram(data_set1)
        for counter in range(B):
            batch = np.random.uniform(0, 1, nu)
            statistics1.append(statistic(tree1, batch))
        if not N2 == np.inf:
            data_set2 = np.random.uniform(0, 1, data_number)
            tree2 = qt.QuantTree(pi_values)
            tree2.build_histogram(data_set2)
            for counter in range(B):
                batch = np.random.uniform(0, 1, nu)
                statistics2.append(statistic(tree2, batch))
        else:
            alternative_computation = aux.Alternative_threshold_computation(pi_values, nu, statistic)
            tree2 = alternative_computation.compute_cut()
            for counter in range(B):
                batch = np.random.uniform(0, 1, nu)
                statistics2.append(statistic(tree2, batch))
        statistics1 = np.sort(statistics1)
        statistics2 = np.sort(statistics2)
        def_values1 = def_values1 + statistics1
        def_values2 = def_values2 + statistics2
    def_values1 = def_values1 / to_average
    def_values2 = def_values2 / to_average
    #PLOT
    # evaluate the histogram
    values1, base1 = np.histogram(def_values1, len(def_values1))
    values2, base2 = np.histogram(def_values2, len(def_values2))
    # evaluate the cumulative
    cumulative1 = np.cumsum(values1)
    cumulative2 = np.cumsum(values2)
    # plot the cumulative function
    plt.plot(base1[:-1], cumulative1, label='cumulative1')
    plt.plot(base1[:-1], cumulative2, label = 'cumulative2 = inf')
    plt.legend()
    plt.title(str(N1) + ' data')
    plt.show()
    return

def test_thresholds_association():
    helper = aux.DataSet_for_the_learner(bins_number, data_number)
    helper.create_multiple_bins_combinations()
    helper.asymptotically_associate_thresholds(nu, statistic, alpha, B)
    for index in range(len(helper.asymptotyical_thresholds)):
        asy = aux.Alternative_threshold_computation(helper.histograms[index], nu, statistic)
    helper.associate_thresholds(200, nu, statistic, alpha, B)
    helper.associate_thresholds(50, nu, statistic, alpha, B)
    print(helper.thresholds)


def check_net():
    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                       alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                       data_number_for_learner)
    test.create_training_set_for_QT()
    test.create_and_train_net()
    for index in range(20):
        histogram = aux.create_bins_combination(bins_number)
        print ('predicted value')
        print(test.learner.predict_threshold(histogram, data_number))
        tree = qt.QuantTree(histogram)
        tree.build_histogram(test.data_set)
        test2 =qt.ChangeDetectionTest(tree, nu, statistic)
        print ('Tru threshold is:')
        print(test2.estimate_quanttree_threshold(alpha, B))

def check_alternative_computaton():
    values2 = []
    alternatives = []
    for index in range(20):
        bins = aux.create_bins_combination(bins_number)
        alt1 = aux.Alternative_threshold_computation(bins, nu, statistic)
        alt = alt1.compute_threshold(alpha, B)
        alternatives.append(alt)
        tree = qt.QuantTree(bins)
        tree.ndata = 30
        norm1 = qt.ChangeDetectionTest(tree, nu, statistic)
        norm = norm1.estimate_quanttree_threshold(alpha, B)
        values2.append(norm)
    plt.title('Box plots of  thresholds')
    plt.boxplot(alternatives, showfliers=False)
    plt.title('Altetnatives')
    plt.show()
    plt.title('Normale, 30 data')
    plt.boxplot(values2, showfliers=False)
    plt.show()
    return

def plot_normal_computation():
    normals = []
    for index in range(5):
        bins = aux.create_bins_combination(bins_number)
        tree = qt.QuantTree(bins)
        tree.ndata = 200000
        test = qt.ChangeDetectionTest(tree, nu, statistic)
        alt = test.estimate_quanttree_threshold(alpha, B)
        normals.append(alt)
        plt.boxplot(normals)
        plt.title('Box plot of normal thresholds')
        plt.legend()
        plt.show()
    return

def compare_power_with_asymptotic():
    number_of_tests_for_the_plot = 50
    normal_to_plot = []
    modified_to_plot = []
    number_of_batches_per_test = 2000

    test = aux.Superman(percentage, SKL, initial_pi_values, data_number,
                        alpha, bins_number, data_Dimension, nu, B, statistic, N_values,
                        data_number_for_learner)
    for index in range(number_of_tests_for_the_plot):
        test.create_training_set_for_QT()
        normal_value = 0
        modified_value = 0
        normal_value += test.run_normal_algorithm(number_of_batches_per_test, False)
        modified_value += test.run_asymtpotic_algorithm_without_learner(number_of_batches_per_test, False)
        normal_to_plot.append(normal_value)
        modified_to_plot.append(modified_value)

    normal_to_plot = np.ones(len(normal_to_plot)) - normal_to_plot
    modified_to_plot = np.ones(len(modified_to_plot)) - modified_to_plot

    print('Powers: modified and normal')
    print( np.mean(modified_to_plot), np.mean(normal_to_plot))
    fig, axs = plt.subplots(2)
    axs[0].boxplot(normal_to_plot, notch=False)
    axs[0].set_title('Standard power')
    axs[1].boxplot(modified_to_plot, notch=False)
    axs[1].set_title('Extended power with asymptotic')
    plt.show()


#plot_cumulative_functions(1000, np.inf, B, initial_pi_values, nu, statistic)


def test_the_powers():
    normals = []
    modified = []
    x_values = []
    for sK in range(10):
        SKL = sK + 1
        norm, ext = compare_power()
        normals.append(norm)
        modified.append(ext)
        x_values.append(sK + 1)

    normal_to_plot = []
    modified_to_plot = []
    for index in range(len(normals)):
        normal_to_plot.append(np.mean(normals[index]))
        modified_to_plot.append(np.mean(modified[index]))
    plt.plot(x_values, normal_to_plot)
    plt.title('Normal powers')
    plt.plot(x_values, modified_to_plot)
    plt.title('Modified values')
    plt.show()
    return

test_the_powers()

#compare_FP0()

#compare_power()
compare_FP0_with_asymptotic()
"""
compare_power_with_asymptotic()


compare_regressor_FP0()
compare_regressor_power()

check_alternative_computaton()
"""