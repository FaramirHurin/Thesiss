import extendedQuantTree as aux
import EWMA_QuantTree
import qtLibrary.libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt

bins_number = 16
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 2000
alpha = [0.5]
beta = 0.1
lambd = 0.01
data_Dimension = 5
nu = 32
B = 4000
X = [3]
max_N = 3000
min_N = nu
SKL = 1


#Dynamic QuantTree tests

def run_normally(run_lenght, beta, statistic):
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(500)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, lambd, statistic, max_N, 100, alpha, False, False, nu, beta)
    x.initialize(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.classic_batch_analysis(batch)
    return np.mean(statistics)

#Plots the EWMA value of one run.
def test_EWMA_once(run_lenght, beta, statistic):
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(500)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, lambd, statistic, max_N, 100, alpha, False, False, nu, beta)
    x.initialize(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.compute_EMWA(batch)
    return statistics, np.mean(x.record_history)

# EMWA threshold is correct. What does it mean?
#Taking random points of independent runs the EMWA is higher than the threshold with probability alpha

#Averages over multiple runs. We expect the EWMA to be around alpha for every time T
def test_EWMA(experiments_number, run_lenght, beta, statistic):
    statistics = np.zeros(run_lenght)
    record_history = np.zeros(run_lenght)
    for exp in range(experiments_number):
        to_add, record_history_to_add =  test_EWMA_once(run_lenght, beta, statistic)
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
    data = data_handler.return_equal_batch(500)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree\
        (initial_pi_values, lambd, statistic, max_N, 100, alpha, False, False, nu, beta)
    x.initialize(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.compute_EMWA(batch)
        if x.find_change():
            break
    return round

model = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, lambd, qt.tv_statistic, max_N, 100, alpha, False, False, nu, beta)
data_handler = aux.Data_set_Handler(data_Dimension)
data = data_handler.return_equal_batch(2000)
model.initialize(data)
stat = np.array(model.alterative_EWMA_thresholds_computation())
for index in range(1, 10):
    stat += np.array(model.alterative_EWMA_thresholds_computation())
stat = stat/index
plt.plot(stat)
plt.title('thresholds')
plt.show()

