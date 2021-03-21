from main_code.algorithms.incremental_QuantTree import Online_Incremental_QuantTree
from main_code.algorithms.EWMA_QuantTree import Offline_EWMA_QuantTree, Online_EWMA_QUantTree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import qtLibrary.libquanttree as qt
import pickle
import matplotlib.pyplot as plt
import numpy as np

dimensions_number = 3
bins_number = 8
initial_pi_values = create_bins_combination(bins_number)
nu = 32
training_N = 5 * bins_number
lamb = 0.2 #Important: can't change val
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

