import main_code.algorithms.incremental_QuantTree as iqt
from sklearn.metrics import r2_score
import logging
from main_code.auxiliary_project_functions import Data_set_Handler
import pickle
import qtLibrary.libquanttree as qt

logger = logging.getLogger('Logging_data_creation')
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
N = 3
DIMENSIONS = 3
NU = 32
BINSNUMBER = 32
ALPHA = 0.02 # ARL0 = 50 (*32)

logging.debug('FPR should be ' + str(ALPHA))
logging.debug('Initial training set size = ' + str(N*NU))

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

assess_normal_net()
