from main_code.algorithms.incremental_QuantTree import Online_Incremental_QuantTree, Incremental_Quant_Tree
from main_code.algorithms.EWMA_QuantTree import Offline_EWMA_QuantTree, Online_EWMA_QUantTree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
from qtLibrary.qt_EWMA_Luca import H_CPM
import main_code.neuralNetworks as nn
import qtLibrary.libquanttree as qt
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


#Parameters
percentage = 0.01
N = 5000
nu = 32
dimensions_number = 3
alpha = [0.01]
bins_number = 8
statistic = qt.tv_statistic


B = 8000
DOES_PICKLE = False


#Control over FPR of the Incremental QuantTree
def compare_FPR_once(percentage, N, nu, QT, IQT, data_set_handler, alpha, run_lenght, statistic, learner):

    training_set =  data_set_handler.return_equal_batch(N)
    initial_training_set = training_set[:int(N*percentage)]
    incremental_training_set = training_set[int(N*percentage):]
    QT.build_histogram(training_set)
    IQT.build_histogram(initial_training_set)
    IQT.modify_histogram(incremental_training_set)
    qt_detector = qt.ChangeDetectionTest(QT, nu, statistic)
    qt_threshold = qt_detector.estimate_quanttree_threshold(alpha, B)

    percentages = np.zeros(2)

    for index in range(run_lenght):
        batch = data_set_handler.return_equal_batch(nu)

        percentages[0] += statistic(QT, batch) > qt_threshold
        percentages[1] = statistic(IQT, batch) > learner.predict_value(IQT.pi_values, IQT.ndata)

    return percentages/run_lenght

def compare_FPR(percentage, N, nu, alpha, dimensions_number, experiments, run_lenght, bins_number, statistic):

    #Initialization
    data_set_handler = Data_set_Handler(dimensions_number)
    to_plot = np.zeros([experiments, 2])
    if DOES_PICKLE:
        with open('../learner_dataset/network.pickle', 'rb') as file:
            learner = pickle.load(file)
    else:
        learner = nn.NN_man(bins_number, 10000, nu, 30)
        learner.train(alpha)
    #Cycle
    for index in range(experiments):
        pi_values = create_bins_combination(bins_number)
        qTree = qt.QuantTree(pi_values)
        iqTree = Incremental_Quant_Tree(pi_values)
        to_plot[index] = compare_FPR_once(percentage, N, nu, qTree, iqTree, data_set_handler, alpha, run_lenght, statistic, learner)

    return to_plot

def test_on_FPR():
    #Set parameters
    experiments = 2
    run_lenght = 10000

    #Run FPR
    to_plot = compare_FPR(percentage, N, nu, alpha, dimensions_number, experiments, run_lenght, bins_number, statistic)

    #Plotting of the result
    normals = to_plot[:, 0]
    incrementals = to_plot[:, 1]
    plt.boxplot([normals, incrementals], labels=['QuantTree', 'Incremental QuantTree'], showfliers=False, showmeans=True)
    plt.title('FPR: alpha = ' + str(alpha[0]))
    plt.show()

    return

#Comparison between power of Incremental and QuantTree

def compare_power_once(percentage, N, nu, QT, IQT, data_set_handler, alpha, run_lenght, statistic, learner, skl):

    training_set =  data_set_handler.return_equal_batch(N)
    initial_training_set = training_set[:int(N*percentage)]
    incremental_training_set = training_set[int(N*percentage):]
    QT.build_histogram(training_set)
    IQT.build_histogram(initial_training_set)
    IQT.modify_histogram(incremental_training_set)
    qt_detector = qt.ChangeDetectionTest(QT, nu, statistic)
    qt_threshold = qt_detector.estimate_quanttree_threshold(alpha, B)

    percentages = np.zeros(2)

    for index in range(run_lenght):
        batch = data_set_handler.generate_similar_batch(nu, skl)

        percentages[0] += statistic(QT, batch) > qt_threshold
        percentages[1] = statistic(IQT, batch) > learner.predict_value(IQT.pi_values, IQT.ndata)

    return percentages/run_lenght

def compare_power(percentage, N, nu, alpha, dimensions_number, experiments, run_lenght, bins_number, statistic, skl):

    #Initialization
    data_set_handler = Data_set_Handler(dimensions_number)
    to_plot = np.zeros([experiments, 2])
    if DOES_PICKLE:
        with open('../learner_dataset/network.pickle', 'rb') as file:
            learner = pickle.load(file)
    else:
        learner = nn.NN_man(bins_number, 10000, nu, 30)
        learner.train(alpha)
    #Cycle
    for index in range(experiments):
        pi_values = create_bins_combination(bins_number)
        qTree = qt.QuantTree(pi_values)
        iqTree = Incremental_Quant_Tree(pi_values)
        to_plot[index] = compare_power_once(percentage, N, nu, qTree, iqTree, data_set_handler, alpha, run_lenght, statistic, learner, skl)

    return to_plot

def test_on_power():
    #Set parameters
    experiments = 2
    run_lenght = 10000
    SKL = [0, 0.1, 0.2, 0.5, 1]

    table = np.zeros([experiments, len(SKL)])
    table2 = np.zeros([experiments, len(SKL)])
    frame_qt = pd.DataFrame(table, columns=[elem for elem in SKL])
    frame_iqt =  pd.DataFrame(table, columns=[elem for elem in SKL])

    #Run power
    for skl in SKL:
        for index in range(experiments):
             x, y= compare_power(percentage, N, nu, alpha, dimensions_number, experiments, run_lenght, bins_number, statistic, skl)
             frame_qt.loc[index, skl] = x
             frame_iqt.loc[index, skl] = y
    #TODO Decide how to plot
    return

# IQC ARLO control

def test_IQC_ARL_once(dataset_handler, tree, skl, max_run_lenght, training_size):
    training_set = dataset_handler.return_equal_batch(int(N*percentage))
    assert len(training_set) > nu
    done = False
    while not done:
        try:
            tree.build_histogram(training_set)
            done = True
        except:
            pi_values = create_bins_combination(bins_number)
            tree = Online_Incremental_QuantTree(pi_values, alpha, statistic)

            done = False

    for round in range(training_size):
        batch = dataset_handler.return_equal_batch(nu)
        tree.tree.modify_histogram(batch)

    for round in range(training_size, max_run_lenght):
        if skl == 0:
            batch = dataset_handler.return_equal_batch(nu)
        else:
            batch = dataset_handler.generate_similar_batch(nu, skl)
        stop = tree.play_round(batch)
        if stop:
            break
    return round

def test_IQC_ARL0():
    experiments = 200
    max_run_lenght = 1000
    training_size = 10
    SKL = [0, 0.1, 0.2, 0.5, 1]
    handler = Data_set_Handler(dimensions_number)
    pi_values = create_bins_combination(bins_number)
    tree = Online_Incremental_QuantTree(pi_values, alpha, statistic)
    table = np.zeros([experiments, len(SKL)])
    frame = pd.DataFrame(table, columns=[elem for elem in SKL])
    for skl in SKL:
        for index in range(experiments):
            frame.loc[index, skl] = test_IQC_ARL_once(handler, tree, skl, max_run_lenght, training_size)
    to_plot = []
    for skl in SKL:
        to_plot.append(frame.loc[:, skl])
    plt.boxplot(to_plot, labels=SKL, showfliers=False, showmeans=True)
    plt.title('Run lenght as function of skl')
    plt.show()
    return

test_IQC_ARL0()

def compare_IQR_ARL1_once():
    arl1 = [0, 0]
    return arl1

def compare_IQR_ARL1():
    arl1_iqt, arl1_HCPM = compare_IQR_ARL1_once()
    return

# EWMA ad P tend to the FPR of the tree oserved (test once with QT, once with IQT)
'''

Run EWMA. Plot the FPR of the algorithm as a straight line, plot Z and P. 
Show that Z and P tend towards the FPR.
Repeat for offline and online.

'''
def return_EWMA_statistics(tree, run_lenght, change_time, skl, training_set_size):

    dataset_handler = Data_set_Handler(dimensions_number)
    training_set = dataset_handler.return_equal_batch(training_set_size)
    tree.build_histogram(training_set)

    for index in range(change_time):
        batch = dataset_handler.return_equal_batch(nu)
        tree.play_round(batch)
    for index in range(run_lenght - change_time):
        if skl == 0:
            batch = dataset_handler.return_equal_batch(nu)
        else:
            batch = dataset_handler.generate_similar_batch(nu, skl)
        tree.play_round(batch)

    if skl == 0:
        mean = np.mean(tree.record_history)
    else:
        mean = [np.mean(tree.record_history[:change_time]), np.mean(tree.record_history[change_time:])]

    return tree.values, tree.FPR_estimator,mean

def plot_EWMA_statistic(online, run_lenght, change_time, skl):
    training_set_size = 100
    pi_values = create_bins_combination(bins_number)
    lamb = 0.1
    alpha = [0.5]
    desired_ARL0 = 100

    if online:
        tree = Online_EWMA_QUantTree(pi_values, lamb, statistic, alpha, nu, desired_ARL0)
    else:
        tree = Offline_EWMA_QuantTree(pi_values, lamb, statistic, alpha, nu, desired_ARL0)

    values, estimators, FPR = return_EWMA_statistics(tree, run_lenght, change_time, skl, training_set_size)

    if skl == 0:
        FPR_to_plot = np.ones(run_lenght) * FPR
    else:
        array1 = np.ones(change_time) * FPR[0]
        array2 = np.ones(run_lenght - change_time) * FPR[1]
        FPR_to_plot = np.append(array1, array2)

    plt.plot(values) #, labels = ['Z', 'P', 'FPR']
    plt.plot(estimators)
    plt.plot(FPR_to_plot)
    plt.title('Behaviour') #TODO Write title
    plt.show()


plot_EWMA_statistic(False, 10000, 1000, 0.5)
# When there is a change, the two statistics tend to the new (higher) value, and EWMA does it faster
'''
Run EWMA. 
At time TAU make a change in the underlying data. 
Plot FPR (separated before and after) Z and P around the change.
Repeat for offline and online.
'''


# We can control the ARL0 of EWMA
'''
Plot ARL0 of EWMA.
Repeat for offline and online.

'''

# ARL1 of EWMA much smaller than ARL0
'''
Plot ARL0 and ARL1 and show that it decreases fast.
Repeat for offline and online.
'''