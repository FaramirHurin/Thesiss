import numpy as np
import pandas as pd
from main_code.algorithms.incremental_QuantTree import Incremental_Quant_Tree
import sys
import main_code.auxiliary_project_functions as aux
import qtLibrary.libquanttree as qt

#print(sys.path)

initial_bins = aux.create_bins_combination(8)
nu = 32
N = 300
distra = aux.Data_set_Handler(3)
alpha = [0.01]
statistic = qt.tv_statistic

def generate_training_set_for_Recurrent(initial_bins, nu, N, distra, alpha, statistic):
    INDIPENDENT_SERIES = 1
    SERIES_LENGHT = 3
    X = np.zeros([SERIES_LENGHT * INDIPENDENT_SERIES, len(initial_bins) + 1])
    Y = np.zeros(SERIES_LENGHT * INDIPENDENT_SERIES)
    for serie in range(INDIPENDENT_SERIES):
        tree = Incremental_Quant_Tree(initial_bins)
        training_set = distra.return_equal_batch(N)
        tree.build_histogram(training_set)
        for time in range(SERIES_LENGHT):
            pi = np.sort(tree.pi_values)
            X[time, :len(tree.pi_values)] = pi
            X[time, -1] = tree.ndata
            Y[time] = qt.ChangeDetectionTest(tree, nu, statistic).estimate_quanttree_threshold(alpha, 5000)
            batch = distra.return_equal_batch(nu)
            tree.modify_histogram(batch)
    #Create array from X and Y
    #Store array
    X_Y = np.zeros([SERIES_LENGHT * INDIPENDENT_SERIES, len(initial_bins) + 2])
    X_Y[:, :-1] = X
    X_Y[:, -1] = Y
    frame = pd.DataFrame(X_Y)
    debug = 0
    #frame.to_csv('Training_set_for_recurrent.csv')

generate_training_set_for_Recurrent(initial_bins=initial_bins, nu=nu, distra=distra, alpha=alpha, statistic=statistic, N=N)