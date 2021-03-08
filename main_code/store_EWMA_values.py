#Imports

# My code
from main_code.EWMA_QuantTree import Offline_EWMA_QuantTree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
import qtLibrary.libquanttree as qt

#Standard libraries
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

#Initialization

#Standard dimensions
dimensions_number = 3
bins_number = 8
nu = 32

#Trees parameters
training_N = 5 * bins_number
lamb = 0.005
alpha_EWMA = [0.5]
alpha_QT = [0.01]
desired_ARL0 = int(1/alpha_QT[0])
statistic = qt.tv_statistic

# pi_values
pi_values = create_bins_combination(bins_number)

#Dataset
handler = Data_set_Handler(dimensions_number)
training_set = handler.return_equal_batch(training_N)

#Offline Tree creation
offline_tree = Offline_EWMA_QuantTree( pi_values, lamb, statistic, alpha_EWMA, nu, desired_ARL0)

#Histograms building, EWMA thresholds computation and assignation
offline_tree.build_histogram(training_set)
offline_EWMA_thresholds = offline_tree.EWMA_thresholds


plt.plot(offline_EWMA_thresholds)
plt.show()

#Creation data structure
dict = {'Alpha': alpha_EWMA, 'desired_ARL0': desired_ARL0, 'pi_values': pi_values, 'Thresholds': offline_EWMA_thresholds}

#Storage

with open('nets.pickle', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Retrieve file
file2 = open('nets.pickle', 'rb')
new_d = pickle.load(file2)
print(new_d)
file2.close()
