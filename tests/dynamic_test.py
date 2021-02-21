from extendedQuantTree import Online_Incremental_QuantTree
from paired_learner import Paired_Learner
from EWMA_QuantTree import Online_EWMA_QUantTree, Offline_EWMA_QuantTree
from auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import qtLibrary.libquanttree as qt
import numpy as np

dimensions_number = 3
bins_number = 8
initial_pi_values = np.ones(bins_number)/bins_number
nu = 16
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

def compare_ARL0( rounds_number ):
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

    stop_time_trees = {online_EWMA: rounds_number, online_Incremental: rounds_number, offline_EWMA: rounds_number}
    stop_time = {online_EWMA: rounds_number, online_Incremental: rounds_number, h_CPM: rounds_number, offline_EWMA: rounds_number}

    for round in range(rounds_number):
        found = {online_EWMA: 0, online_Incremental: 0, offline_EWMA: 0} #, h_CPM: 0
        batch = handler.return_equal_batch(nu)
        if round == 0:
            batches = batch
        else:
            batches = batches + batch
        for algorithm in found:
            if stop_time_trees[algorithm] > round:
                found[algorithm] = algorithm.play_round(batch)
            if found[algorithm]:
                stop_time_trees[algorithm] = round
    for key in stop_time_trees:
        stop_time[key] = stop_time_trees[key]
    stop_time[h_CPM] = h_CPM.test(batches, alpha_QT[0]*nu)[3]
    return stop_time

print(compare_ARL0(1000))