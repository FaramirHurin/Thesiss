import extendedQuantTree as aux
import EWMA_QuantTree
import qtLibrary.libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt
import logging, sys
import superman as sup

percentage = 0.1
bins_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
data_number = 200
alpha = [0.5]
beta = 0.5
data_Dimension = 3
nu = 32
B = 4000
statistic = qt.pearson_statistic
X = [3]
max_N = 3000
min_N = nu
SKL = 1


#Dynamic QuantTree tests

#Plots the EWMA value of one run.
def test_EWMA_once(run_lenght, beta):
    data_handler = aux.Data_set_Handler(data_Dimension)
    data = data_handler.return_equal_batch(nu * 2)
    statistics = np.zeros(run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree(initial_pi_values, 0.1, qt.tv_statistic, 1000, 100, alpha, False, False, nu, beta)
    x.initialize(data)
    for round in range(run_lenght):
        batch = data_handler.return_equal_batch(nu)
        statistics[round] = x.compute_EMWA(batch)
    return statistics

# EMWA threshold is correct. What does it mean?
#Taking random points of independent runs the EMWA is higher than the threshold with probability alpha

#Averages over multiple runs. We expect the EWMA to be around alpha for every time T
def test_EWMA(experiments_number, run_lenght, beta):
    statistics = np.zeros(run_lenght)
    for exp in range(experiments_number):
        to_add =  test_EWMA_once(run_lenght, beta)
        statistics = statistics + to_add
        logging.debug('statistics of current run' + str(statistic))
    statistics = statistics/experiments_number
    plt.plot(range(len(statistics)), statistics)
    plt.title('stat value: ' + str(statistic))
    plt.show()

def test_dynamic_ARL0(max_run_lenght, beta):
    data = aux.Data_set_Handler(data_Dimension)
    statistics = np.zeros(max_run_lenght)
    x = EWMA_QuantTree.EWMA_QuantTree\
        (initial_pi_values, 0.1, qt.tv_statistic, 1000, 100, alpha, True, False, nu, beta)
    for round in range(max_run_lenght):
        batch = data.return_equal_batch(nu)
        statistics[round], stop = x.playRound(batch)
        if stop:
            break
    return round

#Checks wether the threshold computed with different distributions is the same
def check_EMWA_threshold_independence():
    x = EWMA_QuantTree.EWMA_QuantTree\
        (initial_pi_values, 0.01, qt.tv_statistic, 1000, 100, alpha, True, False, nu, beta)
    runs = 20
    normal = np.zeros(10000)
    uniform = np.zeros(runs)
    for counter in range(runs):
        normal +=  x.compute_EMWA_threshold_with_a_stat('normal', nu)
        #uniform_batch = np.random.uniform(0, 1, nu)
        #uniform[counter] = x.compute_EMWA_threshold_with_a_stat('uniform', nu)
    normal = normal/runs
    plt.plot(normal)
    plt.title('EMWA with normal')
    plt.show()
    return

def check_static_EMWA_threshold_independence():
    x = EWMA_QuantTree.Static_EMWA_QuantTree(initial_pi_values, 0.01, qt.tv_statistic, 1000, 100, alpha, True, nu, beta)
    runs = 10
    time = 4000
    normal = np.zeros(time)
    #uniform = np.zeros(runs)
    for counter in range(runs):
        normal += x.compute_EMWA_threshold_with_a_stat('normal', time)
        #uniform += x.compute_EMWA_threshold_with_a_stat('uniform', nu, 1000)
    #plt.boxplot([normal, uniform], labels = ['Normal', 'uniform'])
    normal = normal/runs
    plt.plot(normal)
    plt.title('EMWA static with normal')
    plt.show()
    print('ciao')
    return

"""
x = dynamicQuantTree.Static_EMWA_QuantTree\
    (initial_pi_values, 0.01, qt.tv_statistic, 1000, 100, alpha, True, nu, beta)
for counter in range(10):
    y = x.tell_averaggio('normal', 4000)
    print(y, alpha)
"""
superman = sup.Superman(percentage, SKL, initial_pi_values, data_number,
                             alpha, bins_number, data_Dimension, nu, B, statistic, max_N,
                             50000)


test_EWMA(100, 10, 0.3)