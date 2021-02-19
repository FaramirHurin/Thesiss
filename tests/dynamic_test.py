from extendedQuantTree import Online_Incremental_QuantTree
from paired_learner import Paired_Learner
from EWMA_QuantTree import Online_EWMA_QUantTree
from auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import qtLibrary.libquanttree as qt

dimensions_number = 3
bins_number = 8
initial_pi_values = create_bins_combination(bins_number)
nu = 16
training_N = 4 * bins_number
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
    h_CPM = H_CPM()

    online_EWMA.build_histogram(training_set)
    online_Incremental.build_histogram(training_set)

    stop_time = {online_EWMA: rounds_number, online_Incremental: rounds_number} #, h_CPM: rounds_number

    for round in range(rounds_number):
        found = {online_EWMA: 0, online_Incremental: 0} #, h_CPM: 0
        batch = handler.return_equal_batch(nu)
        for algorithm in found:
            if stop_time[algorithm] > round:
                found[algorithm] = algorithm.play_round(batch)
            if found[algorithm]:
                stop_time[algorithm] = round
    return stop_time

compare_ARL0(300)