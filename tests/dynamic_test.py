from main_code.incremental_QuantTree import Online_Incremental_QuantTree
from main_code.EWMA_QuantTree import Offline_EWMA_QuantTree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import qtLibrary.libquanttree as qt
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

dimensions_number = 3
bins_number = 16
initial_pi_values = create_bins_combination(bins_number)
nu = 32
training_N = 5 * bins_number
lamb = 0.05
alpha_EWMA = [0.3]
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

    #online_EWMA = Online_EWMA_QUantTree(initial_pi_values, lamb, statistic, alpha_EWMA, nu, ewma_desired_ARL0)
    #online_Incremental = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    offline_EWMA = Offline_EWMA_QuantTree(create_bins_combination(bins_number), lamb, statistic, alpha_EWMA, nu, ewma_desired_ARL0)

    h_CPM = H_CPM()
    h_CPM.get_params(dimensions_number, 200)

    #online_EWMA.build_histogram(training_set)
    #online_Incremental.build_histogram(training_set)
    offline_EWMA.build_histogram(training_set)
    final_stop_time = { offline_EWMA.__class__: 0}
    #online_EWMA.__class__: 0,
    #online_Incremental.__class__: 0
    # ,h_CPM.__class__: 0

    for index in range(10):
        stop_time_trees = {
                           offline_EWMA.__class__: rounds_number}  # online_EWMA.__class__: rounds_number,
        #online_Incremental.__class__: rounds_number,
        stop_time = { h_CPM.__class__: rounds_number, offline_EWMA.__class__: rounds_number}
        #online_Incremental.__class__: rounds_number,
        #online_EWMA.__class__: rounds_number,
        values = []
        for round in range(rounds_number):
            found = {offline_EWMA: 0}  # online_EWMA: 0, h_CPM: 0
            #online_Incremental: 0,
            batch = handler.return_equal_batch(nu)
            # Batches = all the batches observed. h_CPM needs to see all the sequence at once
            if round == 0:
                batches = batch
            else:
                batches = batches + batch
            for algorithm in found:
                # Stop time is the minimum between the max run lenght and the actual time stop
                if stop_time_trees[algorithm.__class__] > round:
                    found[algorithm] = algorithm.play_round(batch)
                if found[algorithm]:
                    stop_time_trees[algorithm.__class__] = round
        plt.plot(offline_EWMA.values)
        plt.title('EWMA values during time: stop time is ' + str(stop_time_trees[offline_EWMA.__class__]))
        plt.show()
        print('Percentage accepted by EWMA underlying tree: ' +
                str(np.mean(np.array((offline_EWMA.record_history)))))
        for key in stop_time_trees:
            final_stop_time[key] += stop_time_trees[key]
    for key in stop_time_trees:
        stop_time[key] = stop_time[key] / 10
    #stop_time[h_CPM.__class__] = h_CPM.test(batches, alpha_QT[0]*nu)[3]
    print(final_stop_time)
    return final_stop_time

def run_ARL0():
    dictionary = compare_ARL0(2000)
    runs = 10
    for index in range(runs):
        print('Index is: ' + str(index))
        dictionary = dict(Counter(dictionary) + Counter(compare_ARL0(600)))
    for elem in dictionary:
        dictionary[elem] = dictionary[elem] / (runs + 1)
    print(dictionary)

run_ARL0()

"""Goal: Highlight the problems of the algorithm.
SOLUTIONS:

EWMA: Plot the percentage of batches accepted during a run. It should be 1/desired_ARLO
If this result is reached, plot the EWMA during time. .

Incremental QuantTree: Plot the percentage of batches accepted during a run and verify it is alpha

H_TL: Understand how it works
"""

def chech_ext_ARL0():
    alpha = [0.01]
    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)
    training_set2 = handler.return_equal_batch(training_N)
    batches = [handler.return_equal_batch(nu) for index in range(40)]
    different_batches = [handler.generate_similar_batch(nu, 1) for index in range(1000)]
    tree = Online_Incremental_QuantTree(initial_pi_values, alpha, statistic)
    tree2 = qt.QuantTree(initial_pi_values)
    tree.build_histogram(training_set)
    tree2.build_histogram(training_set2)
    refuses = 0
    refuses2 = 0
    time = 0
    to_print = None
    to_print2 = None
    threshold2 = qt.ChangeDetectionTest(tree2, nu, statistic).estimate_quanttree_threshold(alpha, 6000)
    for batch in batches:
        time += 1
        if tree.play_round(batch):
            refuses += 1
            if to_print is None:
                to_print = time
        if statistic(tree2, batch) > threshold2:
            refuses2 += 1
            if to_print2 is None:
                to_print2 = time
    to_print = None
    to_print2 = None
    for batch in different_batches:
        time += 1
        if tree.play_round(batch):
            refuses += 1
            if to_print is None:
                to_print = time
        if statistic(tree2, batch) > threshold2:
            refuses2 += 1
            if to_print2 is None:
                to_print2 = time
    return(to_print, to_print2)

"""print('First 40 points come from normal')
onl = np.zeros(20)
off = np.zeros(20)
for index in range(20):
    print(index)
    onl[index], off[index]=chech_ext_ARL0()
plt.boxplot([onl, off], labels=['Online', 'Offline'], showfliers=False, showmeans=True)
plt.title('40 normal + modifed')
plt.show()
"""


