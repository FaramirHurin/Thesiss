import pandas as pd
from main_code.algorithms.incremental_QuantTree import Online_Incremental_QuantTree, Incremental_Quant_Tree
from main_code.auxiliary_project_functions import create_bins_combination, Data_set_Handler
from main_code.auxiliary_project_functions import  Alternative_threshold_computation
import qtLibrary.libquanttree as qt
import pickle
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


ALPHA = [0.02]
NU = 32
BINS_NUMBER = 32
INDEPENDENT_RUNS = 30
MAX_TIME = 90 #
DIMENSIONS = 8
DATA_NUMBER = int((INDEPENDENT_RUNS + 1) * MAX_TIME * NU * 1.1)
STATISTIC = qt.tv_statistic
ASYMPTOTOC_SIZE = 1000
INITIAL_BINS = create_bins_combination(BINS_NUMBER)


def record_run(max_time: int, stationary_data: tuple, nu: int, statistic, alpha: list, bins_number: int):
    return_array = []
    thresholds = []
    N = []
    incremental_QuantTree = Incremental_Quant_Tree(INITIAL_BINS)
    for time in range(max_time):
        if time % 10 == 0:
            logger.debug('Iteration number: ' + str(time))
            logger.debug('ndata = ' + str(incremental_QuantTree.ndata))
        if time == 0:
            start_data = stationary_data[: 3 * bins_number]
            incremental_QuantTree.build_histogram(start_data)
            thr = qt.ChangeDetectionTest(incremental_QuantTree, nu, statistic).\
                estimate_quanttree_threshold(alpha, 3000)
        else:
            return_array.append(tuple(incremental_QuantTree.pi_values))
            thresholds.append(thr)
            N.append(incremental_QuantTree.ndata)
            arriving_data = stationary_data[time * nu : time * nu + nu]
            incremental_QuantTree.modify_histogram(arriving_data)
            thr = qt.ChangeDetectionTest(incremental_QuantTree, nu, statistic).\
                estimate_quanttree_threshold(alpha, 3000)
    return return_array, thresholds, N


def create_training_X(independent_runs: int, max_time: int,  stationary_data: tuple,
                      nu: int, statistic: object, alpha: list, bins_number: int):
    bins_combination = []
    thresholds = []
    for index in range(independent_runs):
        logger.debug('Run number: ' + str(index))
        data_for_this_run = stationary_data[index * nu * max_time: (index + 1) * nu * max_time  ]
        bins_story, thresholds_story, N = record_run\
            (max_time=max_time, stationary_data=data_for_this_run, nu=nu,
             statistic=statistic, alpha=alpha, bins_number=bins_number)
        bins_combination += bins_story
        thresholds += thresholds_story
    return tuple(bins_combination), tuple(thresholds), tuple(N)


def crete_stationary_data(dimensions: int, data_Number: int):
    dataSet_handler = Data_set_Handler(dimensions)
    return dataSet_handler.return_equal_batch(data_Number)

def create_asymptotic_training_X(data_number: int, nu:int, statistic: object,  alpha: float, bins_number: int):
    #alpha = [alpha]
    bins_history = []
    thresholds_history = []
    for index in range(data_number):
        logger.debug('Iteration number ' + str(index))
        bins = create_bins_combination(bins_number)
        bins_history.append(bins)
        threshold = Alternative_threshold_computation(bins, nu, statistic).compute_threshold(alpha, 3000)
        thresholds_history.append(threshold)
    return bins_history, thresholds_history


"""
bins_combination = []
data = tuple(crete_stationary_data(dimensions=DIMENSIONS, data_Number=DATA_NUMBER))
X_data, thresholds, N = create_training_X\
    (independent_runs=INDEPENDENT_RUNS,max_time=MAX_TIME,
      stationary_data=data, nu=NU,
     statistic=STATISTIC, alpha=ALPHA, bins_number=BINS_NUMBER)
logger.debug('Asymptotic')
asymptotic_X, asymptotic_thresholds = create_asymptotic_training_X(data_number=ASYMPTOTOC_SIZE, nu=NU,
     statistic=STATISTIC, alpha=ALPHA, bins_number=BINS_NUMBER)
dictionary = {'Normal Bins Series': X_data, 'Normal thresholds series': thresholds, 'ndata series': N,
              'Asymptotic bins series': asymptotic_X, 'Asymptotic thresholds series': asymptotic_thresholds,
              'alpha': ALPHA, 'nu': NU}

fileObject = open('dictionary_to_learn.pickle','wb')
pickle.dump(dictionary, fileObject)
fileObject.close()
"""

file = open('dictionary_to_learn.pickle', 'rb')
dictio = pickle.load(file)
for index in range(30):
    bins = dictio['Normal Bins Series'][index]
    n = dictio['ndata series'][index]
    tree = qt.QuantTree(bins)
    tree.ndata = n
    test = qt.ChangeDetectionTest(tree, NU, STATISTIC)
    thr = test.estimate_quanttree_threshold(ALPHA, 4000)
    logger.debug('Computed ' + str(str(thr)) + ' saved: ' + str(dictio['Normal thresholds series'][index]))
file.close()
