from main_code.incremental_QuantTree import Online_Incremental_QuantTree
from main_code.EWMA_QuantTree import Offline_EWMA_QuantTree, Online_EWMA_QUantTree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
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
lamb = 0.05
alpha_EWMA = [0.5]
alpha_QT = [0.01]
ewma_desired_ARL0 = int(1/alpha_QT[0])
statistic = qt.tv_statistic

"""
Compare dynamic version of extended QUantTree, dynamic EWMA and H_CPM
ARL0:Compare the average run lenght of the three algorithms
ARL1: Two possible changes, one soon and one late
"""

"""Goal: Highlight the problems of the algorithm.
SOLUTIONS:

EWMA: Plot the percentage of batches accepted during a run. It should be 1/desired_ARLO
If this result is reached, plot the EWMA during time. .

Incremental QuantTree: Plot the percentage of batches accepted during a run and verify it is alpha

H_TL: Understand how it works
"""
def compare_algorithms_ARL(SKL = 0):
    online_EWMA = Online_EWMA_QUantTree(initial_pi_values, lamb, statistic, alpha_EWMA, nu, ewma_desired_ARL0)
    online_Incremental = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    offline_EWMA = Offline_EWMA_QuantTree(create_bins_combination(bins_number), lamb, statistic, alpha_EWMA, nu,
                                          ewma_desired_ARL0)
    htl = H_CPM()
    trees = [online_EWMA, online_Incremental, offline_EWMA]

    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)
    for tree in trees:
        tree.build_histogram(training_set)

    rounds_number = 1000

    stop_time = {online_Incremental.__class__: rounds_number,
                 online_EWMA.__class__: rounds_number, offline_EWMA.__class__: rounds_number,
                 htl.__class__:rounds_number}

    for round in range(1000):
        if SKL == 0 or round < 30:
            batch = handler.return_equal_batch(nu)
        else:
            batch = handler.generate_similar_batch(nu, SKL)
        for tree in trees:
            if tree.change_round is not None:
                stop_time[tree.__class__] = round - 1
                break
            if stop_time[tree.__class__] > round:
                stop = tree.play_round(batch)
                if stop:
                    stop_time[tree.__class__] = round
                    tree.restart()
    print('SKL is:' +str(SKL))
    print(stop_time)
    return stop_time


def run(SKL):
    print('SKL is' + str(SKL))
    dic = compare_algorithms_ARL(SKL)
    for index in range(runs):
        part_dic = compare_algorithms_ARL(SKL)
        for elem in dic:
            dic[elem] += part_dic[elem]
    for elem in dic:
        dic[elem] = dic[elem] / (runs + 1)
    print('SKL is ' + str(SKL) + ' Final dictionary:' + str(dic))

print('Change point = 30')
runs = 10
run(SKL=1)
run(SKL=0)
SKL = 0