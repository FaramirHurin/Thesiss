import main_code.algorithms.incremental_QuantTree as iqt
from sklearn.metrics import r2_score
import logging
from main_code.auxiliary_project_functions import Data_set_Handler
import pickle
import qtLibrary.libquanttree as qt
import matplotlib.pyplot as plt
import numpy as np
from main_code.algorithms.EWMA_QuantTree import EWMA_QUantTree, Online_EWMA_QUantTree
import pandas as pd

logger = logging.getLogger(r'C:\Users\dalun\PycharmProjects\Thesiss')
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.DEBUG,
    datefmt='%Y-%m-%d %H:%M:%S')

"""
1) Verify the accuracy of the normal and asymptotic network. On test set

2) Compare the FPR and power of QuantTree and IQT without the NN.
Possible to create a new method in the other file

3) COmpare the FPR and power  of QuantTree and IQT with the NN (other file)

4) Test IQT on its own:  HERE

5) Compare the ARL0 and ARL1 of QuantTree and IQT: Other file

6) Run the same experoment toghether with H-CPM

7) Plot ARL0 of EWMA


"""

STATISTIC = qt.tv_statistic
TREES_NUMBER = 2000
BINSNUMBER = 32
NU = 32
SKL = (1, 2, 4)
DIMENSIONS = (2, 4, 8)
M = 4 * NU
N = 100 * NU
ALPHA = (0.02, 0.002)
RUN_LENGHT = tuple([20 * int(NU/alpha) for alpha in ALPHA])




logging.debug('FPR should be ' + str(ALPHA))
logging.debug('Initial training set size = ' + str(N*NU))
logger.debug('Run lenght is ' + str(RUN_LENGHT))

file = open('dictionary_to_learn.pickle', 'rb')
dictionary = pickle.load(file)
INITIAL_BINS = dictionary['Normal Bins Series'][0]
file.close()



# 1)
def assess_normal_net():
    net = iqt.Neural_Network()
    tree = iqt.Incremental_Quant_Tree(INITIAL_BINS)
    handler = Data_set_Handler(DIMENSIONS)
    data= handler.return_equal_batch(100 * NU)
    training_set = handler.return_equal_batch(N * NU)
    tree.build_histogram(training_set)
    fast_record = []
    slow_record = []
    for round in range(100):
        batch = data[round * NU: (round +1) * NU]
        fast_record.append(net.predict_value(tree.pi_values, tree.ndata))
        slow_record.append(qt.ChangeDetectionTest(tree, NU, qt.tv_statistic).\
            estimate_quanttree_threshold([ALPHA], 3000))
        logger.debug('Slow = ' + str(slow_record[-1]) + ' Fast = ' + str(fast_record[-1][0]))
        tree.modify_histogram(batch)
    logger.debug(r2_score(fast_record, slow_record))
    logger.debug('Fast record is: ' + str(fast_record))
    logger.debug('Slow record is ' + str(slow_record))
    return


# assess_normal_net()


class Experiments_forest:
    def __init__(self, trees_Number, initial_bins, statistic, alpha):
        logger.debug('Starting to create one forest')
        self.statistic = statistic
        self.incremental_trees = [iqt.Online_Incremental_QuantTree(initial_bins, alpha, self.statistic)
                                  for number in range(trees_Number)]
        self.trees = [qt.QuantTree(initial_bins) for number in range(trees_Number)]
        self.alpha = alpha
        logger.debug('Initialized')
        return

    def run_normally(self, run_lenght, dimensions, training_size, nu):
        logger.debug('Run normally')
        handler = Data_set_Handler(dimensions)
        training_set = handler.return_equal_batch(training_size)
        tr2 = handler.return_equal_batch(training_size * 10)
        for tree in self.trees:
            tree.build_histogram(tr2)
        for tree in self.incremental_trees:
            tree.build_histogram(training_set)
        test = qt.ChangeDetectionTest(self.trees[0], nu, self.statistic)
        threshold = test.estimate_quanttree_threshold([self.alpha], 40000)
        tree_lenghts = dict(zip(self.trees, [run_lenght for tree in self.trees]))
        incremental_lenghts = dict(zip(self.incremental_trees,
                                       [run_lenght for tree in self.incremental_trees]))
        batch = []
        for time in range(run_lenght):
            # logger.debug('Time is ' + str(time))
            datum = handler.return_equal_batch(1)[0]
            if len(batch) == nu:
                batch = np.array(batch)
                for tree in self.trees:
                    if tree_lenghts[tree] > time:
                        if self.statistic(tree, batch) > threshold:
                            tree_lenghts[tree] = time
                for tree in self.incremental_trees:
                    if incremental_lenghts[tree] > time:
                        if tree.play_round(batch):
                            incremental_lenghts[tree] = time
                batch = [datum]
            else:
                batch.append(datum)
        logger.debug('Incremental Lenghts are ' + str(incremental_lenghts))
        return incremental_lenghts, tree_lenghts

    def verify_ARL0(self, run_lenght, dimensions, training_size, nu):
        return self.run_normally(run_lenght, dimensions, training_size, nu)

    def verify_FPR(self, run_lenght, dimensions, training_size, nu):
        logger.debug('Verify FPR')
        incremental_lenghts, tree_lenghts = self.run_normally\
            (run_lenght, dimensions, training_size, nu)
        incremental_lenghts = tuple(incremental_lenghts.values())
        tree_lenghts = tuple(tree_lenghts.values())
        qt_FPR = [1 - (tree_lenghts[index + 1] /tree_lenghts[index])
                  for index in range(len(tree_lenghts)- 1)]
        iqt_FPR = [1 - (incremental_lenghts[index + 1] /incremental_lenghts[index])
                   for index in range(len(incremental_lenghts)- 1)]
        return iqt_FPR, qt_FPR

    def run_with_change(self, normal_lenght, changed_lenght, training_size_small,
                        dimensions, nu, skl):
        logger.debug('Run with change')
        handler = Data_set_Handler(dimensions)
        normal_stream = handler.return_equal_batch(normal_lenght)
        training_set = normal_stream[:training_size_small]
        for tree in self.trees:
            tree.build_histogram(normal_stream)
        for tree in self.incremental_trees:
            tree.build_histogram(training_set)
        test = qt.ChangeDetectionTest(self.trees[0], nu, self.statistic)
        threshold = test.estimate_quanttree_threshold([self.alpha], 40000)
        tree_lenghts = dict(zip(self.trees, [changed_lenght for tree in self.trees]))
        incremental_lenghts = dict(zip(self.incremental_trees,
                                       [changed_lenght for tree in self.incremental_trees]))
        batch = []
        for time in range(training_size_small, normal_lenght):
            if len(batch) == nu:
                for tree in self.incremental_trees:
                    tree.update_model(batch)
                batch = [normal_stream[time]]
            else:
                batch.append(normal_stream[time])
        batch = []
        for time in range(changed_lenght):
            # logger.debug('Time is ' + str(time))
            datum = handler.generate_similar_batch(1, skl)[0]
            if len(batch) == nu:
                batch = np.array(batch)
                for tree in self.trees:
                    if tree_lenghts[tree] > time:
                        if self.statistic(tree, batch) > threshold:
                            tree_lenghts[tree] = time
                for tree in self.incremental_trees:
                    if incremental_lenghts[tree] > time:
                        if tree.play_round(batch):
                            incremental_lenghts[tree] = time
                batch = [datum]
            else:
                batch.append(datum)
        return incremental_lenghts, tree_lenghts

    def verify_ARL1(self, normal_lenght, changed_lenght, training_size_small,
                        dimensions, nu, skl):
        return self.run_with_change(normal_lenght, changed_lenght, training_size_small,
                        dimensions, nu, skl)

def test_forest_ARL0():
    ALPHA_USED =  ALPHA[0]
    RUN_LENGHT_USED = RUN_LENGHT[0]
    DIMENSIONS_USED = DIMENSIONS[2]
    forest = Experiments_forest(TREES_NUMBER, INITIAL_BINS, STATISTIC,ALPHA_USED)
    incremental_ARL0, classic_ARL0 = forest.verify_ARL0(RUN_LENGHT_USED, DIMENSIONS_USED, N , NU)
    incremental_ARL0 = tuple(incremental_ARL0.values())
    classic_ARL0 = tuple(classic_ARL0.values())
    logger.debug('Incremental: ' + str(incremental_ARL0) + ' Classic: ' + str(classic_ARL0))
    plt.boxplot([incremental_ARL0, classic_ARL0], labels=['Incremental', 'Classic'], showmeans=True, showfliers=False)
    plt.title('ARL0')
    plt.xlabel('Time: expected = ' + str(NU/ALPHA[0]))
    plt.ylabel('FPR')
    plt.show()
    results = {'alpha' : ALPHA_USED, 'dimensions': DIMENSIONS_USED, 'm': M, 'n': N,
               'run_lenght': RUN_LENGHT_USED, 'IQT ARL0': incremental_ARL0, 'QT_ARL0': classic_ARL0}
    with open('ARL0_file.pickle', 'wb') as file:
        pickle.dump(results, file)
    return

def test_forest_ARL1():
    forest = Experiments_forest(TREES_NUMBER, INITIAL_BINS, STATISTIC, ALPHA)
    incremental_ARL1, classic_ARL1 = \
        forest.verify_ARL1(40 * NU, 6000, N, DIMENSIONS, NU, SKL)
    incremental_ARL1 = tuple(incremental_ARL1.values())
    classic_ARL1 = tuple(classic_ARL1.values())
    logger.debug('Incremental: ' + str(incremental_ARL1) + ' Classic: ' + str(classic_ARL1))
    plt.boxplot([incremental_ARL1, classic_ARL1], labels=['Incremental', 'Classic'],
                showmeans=True, showfliers=False)
    plt.title('ARL1: SKL = ' + str(SKL) + ' D = ' + str(DIMENSIONS))
    plt.xlabel('Time')
    plt.ylabel('Run time')
    plt.show()
    return

# test_forest_ARL1()


# test_forest_ARL0()


def fast_FPR(initial_bins, statistic, alpha, run_lenght, dimensions, training_size, nu):
    incremental = iqt.Online_Incremental_QuantTree(initial_bins, alpha, statistic)
    tree = qt.QuantTree(initial_bins)
    handler = Data_set_Handler(dimensions)
    training_set = handler.return_equal_batch(training_size * nu)
    tree.build_histogram(training_set)
    incremental.build_histogram(training_set)
    tree.build_histogram(training_set)
    test = qt.ChangeDetectionTest(tree, nu, statistic)
    threshold = test.estimate_quanttree_threshold([alpha], 3000)
    batch = []
    incrementally_accepted = 0
    normally_accepted = 0
    for round in range(run_lenght):
        if len(batch) == nu:
            incrementally_accepted += incremental.play_round()
            normally_accepted += statistic(tree, batch) > threshold
    return incrementally_accepted/run_lenght, normally_accepted/run_lenght


def test_fast_FPR():
    inc, nor = fast_FPR(INITIAL_BINS, STATISTIC, ALPHA, RUN_LENGHT, DIMENSIONS, N, NU)
    print('Incremental: ' + str(inc))
    print('Normal' + str(nor))

def test_Forest_FPR():
    ALPHA_USED = ALPHA[0]
    RUN_LENGHT_USED = RUN_LENGHT[0]
    DIMENSIONS_USED = DIMENSIONS[2]
    forest = Experiments_forest(TREES_NUMBER, INITIAL_BINS, STATISTIC, ALPHA_USED)
    incremental_ARL0, classic_ARL0 = forest.verify_ARL0(RUN_LENGHT_USED, DIMENSIONS_USED, N, NU)
    incremental_ARL0 = tuple(incremental_ARL0.values())
    classic_ARL0 = tuple(classic_ARL0.values())
    alive_QT = []
    alive_IQT = []
    for time in range(0, max(max(classic_ARL0), max(incremental_ARL0)), NU):
        a_qt = tuple(filter(lambda x: x > time, classic_ARL0))
        a_iqt = tuple(filter(lambda x: x > time, incremental_ARL0))
        alive_QT.append(len(a_qt))
        alive_IQT.append(len(a_iqt))
    FPR_QT = [1 - alive_QT[index + 1]/alive_QT[index]
              for index in range(len(alive_QT)-1) if alive_QT[index] > 0  ]
    FPR_IQT = [1 - alive_IQT[index + 1]/alive_IQT[index]
                for index in range(len(alive_IQT)-1) if alive_IQT[index] > 0]

    plt.boxplot([FPR_IQT, FPR_QT], labels=['Incremental QuantTree', 'QuantTree'], showmeans=True, showfliers=False)
    plt.title('FPR: alpha = ' + str(ALPHA_USED))
    plt.show()
    return

test_Forest_FPR()

# D: (2, 4, 8)  SKL: (1, 2, 4)  N: 4*NU, M: 100*NU
# FPR, ARL0, ARL1, Power