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
    debug = 0
    for round in range(1000):
        if round == 800:
            debug = 1
            """
            if stop_time[online_EWMA.__class__] == 1000:
                print('Means of late online recognition' + str(np.mean(np.array(online_EWMA.values))))
            if stop_time[offline_EWMA.__class__] == 1000:
                print('Means of late offline recognition' + str(np.mean(np.array(offline_EWMA.values))))
             """
        #Trainhead_tailing phase for ARL1
        if round < 30 and SKL != 0:
            batch = handler.return_equal_batch(nu)
            for tree in trees:
                if tree == online_Incremental:
                    tree.tree.modify_histogram(batch)
                else:
                    tree.modify_histogram(batch)
        #Normal run phase
        else:
            # Batch generation
            if SKL == 0:
                batch = handler.return_equal_batch(nu)
            else:
                batch = handler.generate_similar_batch(nu, SKL)

            for tree in trees:
                if tree.change_round is not None:
                    assert stop_time[tree.__class__] < round
                    continue
                if stop_time[tree.__class__] > round:
                    stop = tree.play_round(batch)
                    if stop:
                        stop_time[tree.__class__] = round

    for key in stop_time:
        results[key].append(stop_time[key])

    for tree in trees:
        tree.restart()
    #print('SKL is:' +str(SKL))
    #print(stop_time)
    assert max(stop_time.values()) < 800 or debug == 1
    return stop_time


def run(SKL, results = None):
    """
    if USES_PICKLE:
        with open(r'..\main_code\nets.pickle', 'rb') as handler:
            dictionary = pickle.load(handler)
            #(dictionary)
            pi_values = dictionary['pi_values']
            ewma_thresholds = dictionary['Thresholds']
            #plt.plot(ewma_thresholds)
            #plt.title('Ewma_thresholds')
            #plt.show()

    """

    pi_values = initial_pi_values

    online_EWMA = Online_EWMA_QUantTree(initial_pi_values, lamb, statistic, alpha_EWMA, nu, ewma_desired_ARL0)
    online_Incremental = Online_Incremental_QuantTree(initial_pi_values, alpha_QT, statistic)
    offline_EWMA = Offline_EWMA_QuantTree(initial_pi_values, lamb, statistic, alpha_EWMA, nu,
                                          ewma_desired_ARL0)
    htl = H_CPM()
    trees = [online_EWMA, online_Incremental, offline_EWMA]
    if results is None:
        results = {online_EWMA.__class__: [], online_Incremental.__class__: []
            , offline_EWMA.__class__: []} #, htl.__class__: []

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
runs = 40
external_runs = 1

print('skl is: ' + str(0))
results0 = run(SKL=0)
for index in range(external_runs):
    results0 = run(SKL = 0, results= results0)
keys = ['online EWMA', 'online Incremental', 'offline EWMA']
values = results0.values()
plt.boxplot(values, labels=keys, showmeans= True, showfliers=False)
plt.title('ARL0')
plt.show()

print('skl is: ' + str(0.5))
results_5 = run(SKL = 0.5)
for index in range(external_runs):
    results_5 = run(SKL=0.5, results=results_5)
values = results_5.values()
plt.boxplot(values, labels=keys, showmeans= True, showfliers=False)
plt.title('ARL1: skl = 0.5')
plt.show()

print('skl is: ' + str(1))
results_1 = run(SKL = 1)
for index in range(external_runs):
    results_01 = run(SKL=1, results=results_1)
values = results_1.values()
plt.boxplot(values, labels=keys, showmeans= True, showfliers=False)
plt.title('ARL1: skl = 1')
plt.show()
