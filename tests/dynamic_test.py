from main_code.incremental_QuantTree import Online_Incremental_QuantTree
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
alpha_QT = [0.01]
statistic = qt.tv_statistic

uses_pickle = True



def compute_ARL0_once(max_time):
    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)

    tree = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    tree.tree.build_histogram(training_set)
    stop_time = max_time
    for round in range(max_time):
        batch = handler.return_equal_batch(nu)
        if tree.play_round(batch):
            stop_time = round
            break
    return stop_time

def compute_ARL(experiments, max_time, SKL = 0, change_time = 0):
    values = []
    for index in range(experiments):
        if SKL == 0:
            values.append(compute_ARL0_once(max_time))
        else: values.append(compute_ARL1_once(max_time, change_time, SKL))
    return values

def compute_ARL1_once(max_time, change_time, SKL):
    handler = Data_set_Handler(dimensions_number)
    training_set = handler.return_equal_batch(training_N)

    tree = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    tree.tree.build_histogram(training_set)
    ARL1 = max_time - change_time
    for round in range(change_time):
        batch = handler.return_equal_batch(nu)
        tree.tree.modify_histogram(batch)
    for round in range(max_time - change_time):
        batch = handler.generate_similar_batch(nu, SKL)
        if tree.play_round(batch):
            ARL1 = round
            break
    return ARL1




values = []
sk = np.linspace(0.5, 2, 4)
for skl in sk:
    values.append(compute_ARL(experiments=300, max_time=4000, SKL=skl, change_time = 20))
plt.boxplot(values, showmeans=True, labels=sk)
plt.title('ARL')
plt.show()



