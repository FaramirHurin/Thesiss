import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import main_code.EWMA_QuantTree as ew
import main_code.auxiliary_project_functions as aux
import qtLibrary.libquanttree as qt
alpha = [0.5]

bins_number = 8
initial_pi_values = aux.create_bins_combination(bins_number)
lamb = 0.01
nu = 32
desired_ARL0 = 100
data_set_size = 1000
dimensions_Number = 3


def einwefe(data, tree):
    tree.restart()
    stop_time = desired_ARL0
    for round in range(desired_ARL0 * 10):
        if tree.change_round is not None:
            stop_time = round - 1
            break
        batch = data.return_equal_batch(nu)
        if tree.play_round(batch):
            stop_time = round
            break
    """if not index%10:
        plt.plot(tree.values)
        plt.plot(tree.EWMA_thresholds)
        plt.title('Values and thresholds')
        plt.show()
    """
    return stop_time

def run(offline):
    data = aux.Data_set_Handler(dimensions_Number)

    tr = data.return_equal_batch(data_set_size)
    
    if offline:
        # Offline
        tree = ew.Offline_EWMA_QuantTree(initial_pi_values=initial_pi_values, lamb=lamb,
                                         statistic=qt.tv_statistic, alpha=alpha, nu=nu,
                                         desired_ARL0=desired_ARL0)
    else:
        tree = ew.Online_EWMA_QUantTree(initial_pi_values=initial_pi_values, lamb=lamb,
                                        statistic=qt.tv_statistic, alpha=alpha, nu=nu,
                                        desired_ARL0=desired_ARL0)

    # Online
    """
   
    """
    tree.build_histogram(tr)

    stop = []
    for index in range(600):
        stop.append(einwefe(data, tree))
    return stop

offline = run(True)
online = run(False)
plt.boxplot([offline, online], labels=['Offline EWMA', 'Online EWMA'], showfliers=False, showmeans=True)
plt.title('ARL0: desired is 100')
plt.show()





