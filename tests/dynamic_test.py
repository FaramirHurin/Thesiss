from main_code.incremental_QuantTree import Online_Incremental_QuantTree
from main_code.EWMA_QuantTree import Offline_EWMA_QuantTree, Online_EWMA_QUantTree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import qtLibrary.libquanttree as qt
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import pickle

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

uses_pickle = True



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
def compare_algorithms_ARL(trees, handler, results, SKL = 0):
    #print('SKL in internal cycle is: ' + str(SKL))
    online_EWMA = trees[0]
    online_Incremental = trees[1]
    offline_EWMA = trees[2]

    rounds_number = 1000

    stop_time = {online_Incremental.__class__: rounds_number,
                 online_EWMA.__class__: rounds_number, offline_EWMA.__class__: rounds_number}

    for round in range(1000):
        if  round < 30 and SKL != 0:
            batch = handler.return_equal_batch(nu)
            for tree in trees:
                if tree == online_Incremental:
                    tree.tree.modify_histogram(batch)
                else:
                    tree.modify_histogram(batch)
        else:
            if SKL == 0:
                batch = handler.return_equal_batch(nu)
            else:
                batch = handler.generate_similar_batch(nu, SKL)
            for tree in trees:
                if tree.change_round is not None:
                    continue
                if stop_time[tree.__class__] > round:
                    stop = tree.play_round(batch)
                    if stop:
                        stop_time[tree.__class__] = round
                        results[tree.__class__].append(round)
    for tree in trees:
        tree.restart()
    #print('SKL is:' +str(SKL))
    #print(stop_time)
    return stop_time


def run(SKL, results = None):
    if uses_pickle:
        with open('..\learner_dataset\pi_valuesANDewma_thresholds.pickle', 'rb') as handler:
            dictionary = pickle.load(handler)
            #(dictionary)
            initial_pi_values = dictionary['pi_values']
            ewma_thresholds = dictionary['Thresholds']
    online_EWMA = Online_EWMA_QUantTree(initial_pi_values, lamb, statistic, alpha_EWMA, nu, ewma_desired_ARL0, ewma_thresholds)
    online_Incremental = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    offline_EWMA = Offline_EWMA_QuantTree(initial_pi_values, lamb, statistic, alpha_EWMA, nu,
                                          ewma_desired_ARL0, ewma_thresholds)
    htl = H_CPM()
    trees = [online_EWMA, online_Incremental, offline_EWMA]
    if results is None:
        results = {online_EWMA.__class__: [], online_Incremental.__class__: []
            , htl.__class__: [], offline_EWMA.__class__: []}

    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)
    for tree in trees:
        tree.build_histogram(training_set)
    dic = compare_algorithms_ARL(trees, handler, results, SKL)
    for index in range(runs):
        part_dic = compare_algorithms_ARL(trees, handler, results, SKL)
        for elem in dic:
            dic[elem] += part_dic[elem]
    for elem in dic:
        dic[elem] = dic[elem] / (runs + 1)
    #print(results)
    #print('SKL is ' + str(SKL) + ' Final dictionary:' + str(dic))
    return results

print('Change point = 30')
runs = 20
external_runs = 5

print('skl is: ' + str(0))
results0 = run(SKL=0)
for index in range(external_runs):
    results0 = run(SKL = 0, results= results0)
print(results0)

print('skl is: ' + str(0.5))
results_5 = run(SKL = 0.5)
for index in range(external_runs):
    results_5 = run(SKL=0.5, results=results_5)
print(results_5)

print('skl is: ' + str(0.1))
results_01 = run(SKL = 0.1)
for index in range(external_runs):
    print('skl is: ' + str(0.1))
    results_01 = run(SKL=0.1, results=results_01)
print(results_01)

for elem in results0:
    print(elem)
    print('Results 0')
    if len(results0[elem]) != 0:
        print(sum(results0[elem])/len(results0[elem]))
    print('Results 0.5')
    if len(results0[elem]) != 0:
        print(sum(results_5[elem])/len(results_5[elem]))
    print('Results 0.1')
    if len(results0[elem]) != 0:
        print(sum(results_01[elem])/len(results_01[elem]))
