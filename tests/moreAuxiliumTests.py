import auxiliary_project_functions
import extendedQuantTree as aux
import qtLibrary.libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt
import neuralNetworks

percentage = 0.1
bins_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 200
alpha = [0.01]
data_Dimension = 3
nu = 32
B = 3000
statistic = qt.pearson_statistic
X = [3]
data_number_for_learner = 10000
max_N = bins_number
min_N = nu
SKL = 1

def test_cuts():
    pi_values = auxiliary_project_functions.create_bins_combination(bins_number)
    pi_values = np.sort(pi_values)
    computation = auxiliary_project_functions.Alternative_threshold_computation(pi_values, nu, statistic)
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
        pi_values = auxiliary_project_functions.create_bins_combination(bins_number)
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
            alternative_computation = auxiliary_project_functions.Alternative_threshold_computation(pi_values, nu, statistic)
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
    helper = neuralNetworks.DataSet_for_the_learner(bins_number, data_number)
    helper.create_multiple_bins_combinations()
    helper.asymptotically_associate_thresholds(nu, statistic, alpha, B)
    for index in range(len(helper.asymptotyical_thresholds)):
        asy = auxiliary_project_functions.Alternative_threshold_computation(helper.histograms[index], nu, statistic)
    helper.associate_thresholds(200, nu, statistic, alpha, B)
    helper.associate_thresholds(50, nu, statistic, alpha, B)
    print(helper.thresholds)

def check_alternative_computaton():
    values2 = []
    alternatives = []
    for index in range(20):
        bins = auxiliary_project_functions.create_bins_combination(bins_number)
        alt1 = auxiliary_project_functions.Alternative_threshold_computation(bins, nu, statistic)
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
        bins = auxiliary_project_functions.create_bins_combination(bins_number)
        tree = qt.QuantTree(bins)
        tree.ndata = 20000
        test = qt.ChangeDetectionTest(tree, nu, statistic)
        alt = test.estimate_quanttree_threshold(alpha, B)
        normals.append(alt)
        plt.boxplot(normals)
        plt.title('Box plot of normal thresholds')
        plt.legend()
        plt.show()
    return

def test_asymptotic_dataSet(alpha):
    nodes = 10000
    n = neuralNetworks.NN_man(bins_number, 40, 20, nodes)
    # n.store_asymptotic_dataSet(data_number_for_learner, nu, statistic, alpha, B)
    histogrms, thresholds = n.retrieve_asymptotic_dataSet(alpha)
    for index in range(len(histogrms)):
        print ('True value ' + str(
            auxiliary_project_functions.Alternative_threshold_computation(histogrms[index], nu, statistic).compute_threshold(alpha, B)))
        print ('Value stored ' + str(thresholds[index]))
