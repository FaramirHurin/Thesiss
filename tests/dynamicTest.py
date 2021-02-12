import extendedQuantTree as aux
import EWMA_QuantTree
import qtLibrary.libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt

bins_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 2000
alpha = [0.5]
desired_ARL0 = 40
lambd = 0.01
data_Dimension = 5
nu = 32
B = 4000
X = [3]
SKL = 1

#Dynamic QuantTree tests

def run_normally(run_lenght, desired_ARL0, statistic):
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(500)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, lambd, statistic, alpha, False, nu, desired_ARL0)
    x.build_histogram(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.classic_batch_analysis(batch)
    return np.mean(statistics)

#Plots the EWMA value of one run.
def test_EWMA_once(run_lenght, desired_ARL0, statistic):
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(500)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, lambd, statistic, alpha, False,  nu, desired_ARL0)
    x.build_histogram(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.compute_EMWA(batch)
    return statistics, np.mean(x.record_history)

# EMWA threshold is correct. What does it mean?
#Taking random points of independent runs the EMWA is higher than the threshold with probability alpha

#Averages over multiple runs. We expect the EWMA to be around alpha for every time T
def test_EWMA(experiments_number, run_lenght, desired_ARL0, statistic):
    statistics = np.zeros(run_lenght)
    record_history = np.zeros(run_lenght)
    for exp in range(experiments_number):
        to_add, record_history_to_add =  test_EWMA_once(run_lenght, desired_ARL0, statistic)
        record_history = record_history + np.array(record_history_to_add)
        statistics = statistics + to_add
        print(record_history_to_add)
    statistics = statistics/experiments_number
    record_history = record_history/experiments_number
    print(np.mean(record_history))
    if statistic == qt.pearson_statistic:
        color = 'r'
    else:
        color = 'b'
    plt.plot(range(len(statistics)), statistics, color)
    plt.title('stat values: r = pearson, b = TV')

def test_ARLO_once(run_lenght, statistic):
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(4000)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree\
        (initial_pi_values, lambd, statistic, alpha, False, nu, desired_ARL0)
    x.build_histogram(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.compute_EMWA(batch)
        if x.find_change():
            break
    return round

def plot_EWMA_thresholds(initial_data_set, experiments):
    model = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, lambd, qt.tv_statistic, alpha, False, nu, desired_ARL0)
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(initial_data_set)
    model.build_histogram(data)
    stat = np.array(model.alterative_EWMA_thresholds_computation())
    for index in range(1, experiments):
        stat += np.array(model.alterative_EWMA_thresholds_computation())
    stat = stat / index
    plt.plot(stat)
    plt.title('thresholds')
    plt.show()


test_EWMA(1, 10 * desired_ARL0, desired_ARL0, qt.tv_statistic)
test_EWMA(1, 10 * desired_ARL0, desired_ARL0, qt.pearson_statistic)
plt.show()

pl = []
for index in range(100):
    print(index)
    pl.append(test_ARLO_once(400, qt.tv_statistic))
plt.boxplot(pl, showmeans=True)
plt.title('ARl0: expected is' + str(desired_ARL0))
print('Mean is: ' + str(np.mean(np.array(pl))))
plt.show()