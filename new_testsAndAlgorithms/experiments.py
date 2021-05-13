import pandas as pd
import numpy as np
from main_code.algorithms.incremental_QuantTree import Online_Incremental_QuantTree, Incremental_Quant_Tree, Neural_Network
from main_code.algorithms.EWMA_QuantTree import Offline_EWMA_QuantTree, Online_EWMA_QUantTree
from main_code.auxiliary_project_functions import Data_set_Handler
import qtLibrary.libquanttree as qt
import pickle
import matplotlib.pyplot as plt
from collections import namedtuple
import logging
from sklearn.metrics import r2_score

logger = logging.getLogger('Logging_data_creation')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')


NU = 32
BINSNUMBER = 32
DIMENSIONS = (3, 8)
N = (92, 1024) #1000
SKL = (1, 2, 4)
MAX_LENGHT = 8000
ALPHA = 0.02 # ARL0 = 50 (*32)
CT = [10, 50 ]
CHANGE_TIME = (tuple(map(lambda x: int(x/ALPHA * NU), CT)))

STATISTIC = qt.tv_statistic
EXPERIMENTS = 20

logging.debug('FPR should be ' + str(ALPHA))

file = open('dictionary_to_learn.pickle', 'rb')
dictionary = pickle.load(file)
INITIAL_BINS_IQT = dictionary['Normal Bins Series'][0]
file.close()
INITIAL_BINS_QT = list(np.ones(BINSNUMBER)/BINSNUMBER)

Setting = namedtuple('Setting', ['skl', 'N', 'dimensions', 'change_time'])

def compare_detection_power_once\
                (initial_pi_values_iqt: list, initial_pi_values_qt:list,  small_TR, bigTR, diff_TR, new_Data, nu:int):
    '''
    :arg  small_TR
    :arg  bigTR
    :arg  new_Data Elements belonging to bigTR but not to smallTR
    :return: percentages of batches accepted by the two algorithms
    '''
    iqt = Incremental_Quant_Tree(initial_pi_values_iqt)
    quantTree = qt.QuantTree(initial_pi_values_qt)
    record_qt = 0
    record_iqt = 0
    quantTree.build_histogram(bigTR)
    iqt.build_histogram(small_TR)
    for index in range(len(diff_TR)):
        iqt.modify_histogram(diff_TR[index * nu: (index + 1) * nu])

    detection_test = qt.ChangeDetectionTest(quantTree, nu, STATISTIC)
    threshold_qt = detection_test.estimate_quanttree_threshold([ALPHA], 5000)
    net = Neural_Network()
    threshold_iqt = net.predict_value(iqt.pi_values, iqt.ndata)

    iterations = int(len(new_Data) / nu)
    for index in range(iterations):
        batch = new_Data[index * nu: (index + 1) * nu]
        stat_value_qt = STATISTIC(quantTree, batch)
        stat_value_iqt = STATISTIC(iqt, batch)
        record_qt += stat_value_qt > threshold_qt
        record_iqt += stat_value_iqt > threshold_iqt

    return record_qt/iterations, record_iqt/iterations


DIMENSIONS_USED = DIMENSIONS[0]
SKL_USED = SKL[1]
def compare_detection_power(initial_pi_values_iqt: list, initial_pi_values_qt:list,
                            smallN: int, bigN: int, nu: int, dimensions_number: int, experiments_N: int,
                            experiment_lenght: int, skl: int):
    handler = Data_set_Handler(dimensions_number)
    small_TR = handler.return_equal_batch(smallN)
    diff_TR = handler.return_equal_batch(bigN - smallN)
    big_TR = np.concatenate((small_TR, diff_TR), axis=0)
    record_qt = []
    record_iqt = []
    new_data = handler.generate_similar_batch(experiment_lenght, skl)
    for exp in range(experiments_N):
        last_qt, last_iqt = compare_detection_power_once(initial_pi_values_iqt, initial_pi_values_qt,
                                                         small_TR, big_TR, diff_TR, new_data, nu)
        record_qt.append(last_qt)
        record_iqt.append(last_iqt)
    return record_qt, list(map(lambda x: x[0], record_iqt))

def test_power():
    qt_to_plot, iqt_to_plot = compare_detection_power(INITIAL_BINS_IQT, INITIAL_BINS_IQT, N[0], N[1], NU,
                                                      DIMENSIONS_USED, 4, 200, SKL_USED)
    logger.debug('QT are' + str(qt_to_plot))
    logger.debug('IQT are' + str(iqt_to_plot))
    plt.boxplot([qt_to_plot, iqt_to_plot], labels=['QuantTree', 'Incremental QuantTree'], showmeans=True)
    plt.title('Detection power: dimensions = ' + str(DIMENSIONS_USED) + ' , SKL = ' + str(SKL_USED))
    plt.show()

test_power()



"""
def run_IQT_once(nu: int, initial_bins: list, alpha: float,
                 training_set: np.ndarray, data_stream: np.ndarray, statistic):
    tree = Online_Incremental_QuantTree(initial_bins, alpha, statistic)
    tree.build_histogram(training_set)
    buffer = []
    spotted = 0
    A_vector = []
    B_vector = []
    for round in range(len(data_stream)):
        if len(buffer) < nu:
            buffer.append(data_stream[round])
        else:
            change = tree.play_round(np.array(buffer))
            # slow_threshold = qt.ChangeDetectionTest(tree.tree, nu, statistic).estimate_quanttree_threshold([alpha], 3000)
            # precise_threshold = qt.ChangeDetectionTest(tree.tree, nu, statistic).estimate_quanttree_threshold([alpha], 7000)
            # fast_threshold = tree.network.predict_value(tree.tree.pi_values, tree.tree.ndata)
            # logger.info('Fast threshold is ' +str(fast_threshold))
            # logger.info('Thresholds are: slow is ' + str(slow_threshold)+ ' fast is ' +
            #               str( fast_threshold[0]))  # + ' slow precise ' + str(precise_threshold)
            buffer = [data_stream[round]]
            # A_vector.append(slow_threshold)
            # B_vector.append(fast_threshold)
            if change:
                spotted += 1
                # break
    # logger.debug('R_2 score is ' + str(r2_score(A_vector, B_vector)))
    return spotted * nu /round #round


def run_QT_once(nu: int, initial_bins: list, alpha: float,
                 training_set: np.ndarray, data_stream: np.ndarray, statistic):
    spotted = 0
    alpha = [alpha]
    tree = qt.QuantTree(initial_bins)
    tree.build_histogram(training_set)
    buffer = []
    detector = qt.ChangeDetectionTest(tree, nu, statistic)
    threshold = detector.estimate_quanttree_threshold(alpha, 8000)
    for round in range(len(data_stream)):
        if len(buffer) < nu:
            buffer.append(data_stream[round])
        else:
            buffer_array = np.array(buffer)
            stat = statistic(tree, buffer_array)
            change = stat > threshold
            buffer = [data_stream[round]]
            if change:
                # break
                spotted += 1
    return spotted * nu / round


def run_HCPM_once(nu: int, alpha: float,
                 training_set: np.ndarray, pre_change_set: np.ndarray, post_change_set: np.ndarray):
    return 0


def compute_single_run_lenght(N: int, nu: int, sKL:float, initial_bins: list, statistic,
                       change_time:int, max_lenght:int, handler:Data_set_Handler, alpha:float):
    training_set = handler.return_equal_batch(N)
    pre_change_set = handler.return_equal_batch(change_time)
    if skl == 0:
        post_change_set =  handler.return_equal_batch(max_lenght - change_time, sKL)
    else:
        post_change_set = handler.generate_similar_batch(max_lenght - change_time, sKL)
    data_stream = np.append(pre_change_set, post_change_set, axis=0)

    qt_result = run_QT_once(nu=nu, initial_bins=initial_bins, alpha=alpha,
                                training_set=training_set, data_stream=data_stream, statistic=statistic)
    iqt_result = run_IQT_once(nu=nu, initial_bins=initial_bins, alpha=alpha,
                                  training_set=training_set, data_stream=data_stream, statistic=statistic)
    # logger.debug('Run lenght IQT = ' + str(run_lenght_IQT))
    # logger.debug('Run lenght QT = ' + str(run_lenght_QT))

    # run_lenght_HCPM = run_HCPM_once()
    return iqt_result, qt_result# , run_lenght_HCPM


def average_run_lenght(N: int, nu: int,  sKL:float, statistic,
                       change_time:int, max_lenght:int, experiments_number: int, alpha:float, handler: Data_set_Handler):
    record_QT = []
    record_IQT = []
    record_HCPM = []
    for index in range(experiments_number):
        iqt_result, qt_result = compute_single_run_lenght\
            (N=N, nu=nu, sKL = sKL, change_time=change_time,
             max_lenght=max_lenght, handler=handler, alpha=alpha, statistic=statistic, initial_bins=INITIAL_BINS)
        # , run_lenght_HCPM
        record_QT.append(qt_result)
        record_IQT.append(iqt_result)
        #record_HCPM.append(run_lenght_HCPM)
    return record_IQT, record_QT #, record_HCPM



keys = []
for skl in SKL:
    for n in N:
        for dimension in DIMENSIONS:
            for change_time in CHANGE_TIME:
                keys.append(Setting(skl, n, dimension, change_time))

dictionary = dict(zip(keys, [None for index in range(len(keys))]))
for key in dictionary.keys():
    if key.skl == 0:
        change = 100000
    else:
        change = key.change_time
    # logger.info('Expected run time = ' + str(min(change, 1 / ALPHA * NU)))

    logger.debug('Key is: ' + str(key))
    handler = Data_set_Handler(key.dimensions)
    record_IQT, record_QT = average_run_lenght(N=key.N, nu=NU, sKL=key.skl, statistic=STATISTIC,
                                         change_time=key.change_time, max_lenght=MAX_LENGHT, alpha=ALPHA, handler=handler, experiments_number=EXPERIMENTS)

    logger.debug('QT Positive rate = ' + str(np.mean(record_QT)) +
                 ' IQT positive rate = ' + str(np.mean(record_IQT)))

"""
"""
handler = Data_set_Handler(8)
change_time = 600
max_lenght = 2000
sKL = 1
training_set = handler.return_equal_batch(96)
pre_change_set = handler.return_equal_batch(change_time)
post_change_set = handler.generate_similar_batch(max_lenght - change_time, sKL)
data_stream = np.append(pre_change_set, post_change_set, axis=0)
run_IQT_once(NU, INITIAL_BINS, ALPHA, training_set, data_stream, STATISTIC)
"""