import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import main_code.EWMA_QuantTree as ew
import main_code.auxiliary_project_functions as aux
import qtLibrary.libquanttree as qt
alpha = [0.5]

bins_number = 16
initial_pi_values = aux.create_bins_combination(bins_number)
lamb = 0.01
nu = 32
desired_ARL0 = 100
data_set_size = 1000
dimensions_Number = 3


def einwefe():
    tree.build_histogram(tr)
    for round in range(desired_ARL0):
        batch = data.return_equal_batch(nu)
        stop = tree.play_round(batch)
        plt.plot(tree.values)
        plt.plot(tree.EWMA_thresholds)
        plt.title('Values and thresholds')
        plt.show()
    return stop


data = aux.Data_set_Handler(dimensions_Number)


tr = data.return_equal_batch(data_set_size)


tree = ew.Offline_EWMA_QuantTree(initial_pi_values=initial_pi_values, lamb = lamb,
                                 statistic = qt.tv_statistic, alpha = alpha, nu = nu,
                                 desired_ARL0=desired_ARL0)
stop = []
for index in range(20):
    stop.append( einwefe())
plt.boxplot(stop)
plt.title('Stop Time')
plt.show()





