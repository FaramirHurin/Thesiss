from extendedQuantTree import Online_Incremental_QuantTree
from paired_learner import Paired_Learner
from EWMA_QuantTree import Online_EWMA_QUantTree, Offline_EWMA_QuantTree
from auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import qtLibrary.libquanttree as qt
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

dimensions_number = 3
bins_number = 8
initial_pi_values = create_bins_combination(bins_number)
nu = 32
training_N = 5 * bins_number
stop = True
lamb = 0.1
alpha_EWMA = [0.5]
alpha_QT = [0.01]
ewma_desired_ARL0 = int(1/alpha_QT[0])
statistic = qt.tv_statistic

"""
Compare dynamic version of extended QUantTree, dynamic EWMA and H_CPM
ARL0:Compare the average run lenght of the three algorithms
ARL1: Two possible changes, one soon and one late
"""

def compare_ARL0( rounds_number):
    '''
    Receives a distribution phi0, and a small training set TR.
    It generates a continuous data flow
    '''
    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)

    online_EWMA = Online_EWMA_QUantTree(initial_pi_values, lamb, statistic, alpha_EWMA, stop, nu, ewma_desired_ARL0)
    online_Incremental = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    offline_EWMA = Offline_EWMA_QuantTree(create_bins_combination(bins_number), lamb, statistic, alpha_EWMA, stop, nu, ewma_desired_ARL0)

    h_CPM = H_CPM()
    h_CPM.get_params(dimensions_number, 200)

    online_EWMA.build_histogram(training_set)
    online_Incremental.build_histogram(training_set)
    offline_EWMA.build_histogram(training_set)

    stop_time_trees = {online_EWMA.__class__: rounds_number, online_Incremental.__class__: rounds_number,
                       offline_EWMA.__class__: rounds_number}
    stop_time = {online_EWMA.__class__: rounds_number, online_Incremental.__class__: rounds_number,
                 h_CPM.__class__: rounds_number, offline_EWMA.__class__: rounds_number}

    for round in range(rounds_number):
        found = {online_EWMA: 0, online_Incremental: 0, offline_EWMA: 0} #, h_CPM: 0
        batch = handler.return_equal_batch(nu)
        #Batches = all the batches observed. h_CPM needs to see all the sequence at once
        if round == 0:
            batches = batch
        else:
            batches = batches + batch

        for algorithm in found:
            #Stop time is the minimum between the max run lenght and the actual time stop
            if stop_time_trees[algorithm.__class__] > round:
                found[algorithm] = algorithm.play_round(batch)
            if found[algorithm]:
                stop_time_trees[algorithm.__class__] = round

    for key in stop_time_trees:
        stop_time[key] = stop_time_trees[key]
    #stop_time[h_CPM.__class__] = h_CPM.test(batches, alpha_QT[0]*nu)[3]
    return stop_time

def run_ARL0():
    dictionary = compare_ARL0(600)
    runs = 30
    for index in range(runs):
        dictionary = dict(Counter(dictionary) + Counter(compare_ARL0(600)))
    for elem in dictionary:
        dictionary[elem] = dictionary[elem] / (runs + 1)
    print(dictionary)


"""Goal: Highlight the problems of the algorithm.
SOLUTIONS:

EWMA: Plot the percentage of batches accepted during a run. It should be 1/desired_ARLO
If this result is reached, plot the EWMA during time. .

Incremental QuantTree: Plot the percentage of batches accepted during a run and verify it is alpha

H_TL: Understand how it works
"""

def verify_EWMA_alpha(rounds_number):
    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)
    online_EWMA = Online_EWMA_QUantTree(initial_pi_values, lamb, statistic, alpha_EWMA, stop, nu, ewma_desired_ARL0)
    online_EWMA.build_histogram(training_set)
    accepted = 0
    for round in range(rounds_number):
        batch = handler.return_equal_batch(nu)
        accepted += online_EWMA.classic_batch_analysis(batch)
        online_EWMA.modify_histogram(batch)
    return float(accepted/rounds_number)

"""
values = [ verify_EWMA_alpha(400) for index in range(10)]
plt.boxplot(values)
plt.title('online EWMA-alpha')
plt.show()
"""

compare_ARL0(500)
incrementals = []
online_EWMA = []
offline_EWMA = []
counter = 20
for index in range(counter):
    print('Experiment number' + str(counter))
    dic = compare_ARL0(1000)
    online_EWMA.append(dic[Online_EWMA_QUantTree])
    offline_EWMA.append(dic[Offline_EWMA_QuantTree])
    incrementals.append(dic[Offline_EWMA_QuantTree])
print([incrementals, online_EWMA, offline_EWMA])
plt.boxplot([incrementals, online_EWMA, offline_EWMA],
            labels=['Incrementals', 'Online', 'Offline'])
plt.title('ARL0')
plt.show()

