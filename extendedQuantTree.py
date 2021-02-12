import numpy as np
import qtLibrary.libquanttree as qt
import qtLibrary.libccm as ccm

# Uses cuts on space instead of the ones on probabilites, equal to normal cut with N=Inf
class Alternative_threshold_computation:

    def __init__(self, pi_values, nu, statistic):
        self.pi_values = pi_values
        self.nu = nu
        self.statistic = statistic
        return

    # Versione modificata di qt.QuantTreeUnivariate.buildHistogram()
    # TODO Rivedere logica leaves
    def compute_cut(self):
        definitive_pi_values = np.zeros(len(self.pi_values))
        histogram = np.zeros(len(self.pi_values)+1)
        bins = []
        interval_still_to_cut = [0, 1]
        left_count = 1
        right_count = 1
        for value in self.pi_values:
            bernoulli_value = np.random.binomial(1, 0.5)
            if bernoulli_value == 0:
                interval_still_to_cut[0] = interval_still_to_cut[0] + value
                histogram[left_count] = interval_still_to_cut[0]
                definitive_pi_values[left_count-1] = value
                left_count +=1
            else:
                interval_still_to_cut[1] = interval_still_to_cut[1] - value
                histogram[-right_count-1] = interval_still_to_cut[1]
                definitive_pi_values[- right_count] = value
                right_count += 1

        histogram = np.transpose(histogram)
        histogram[0] = 0
        histogram[-1] = 1
        self.pi_values = definitive_pi_values
        tree = qt.QuantTreeUnivariate(self.pi_values)
        tree.leaves = histogram
        return tree

    def compute_threshold(self, alpha, B):
        alpha = alpha[0]
        stats = []
        histogram = self.compute_cut()
        for b_count in range(B):
            W = np.random.uniform(0, 1, self.nu)
            thr = self.statistic(histogram, W)
            stats.append(thr)
        stats.sort()
        threshold = stats[int((1-alpha)*B)]
        return threshold

# Extends canonic quantTree with the possibility to modify the histogram associated
class Extended_Quant_Tree(qt.QuantTree):

    def __init__(self, pi_values):
        super().__init__(pi_values)
        self.ndata = 0

    def build_histogram(self, data, do_PCA=False):
        super().build_histogram(data, do_PCA)
        self.ndata = len(data)

    def modify_histogram(self, data, definitive = True):
        self.pi_values = self.pi_values * self.ndata
        bins = self.find_bin(data)
        vect_to_add = np.zeros(len(self.pi_values))
        for index in range(len(self.pi_values)):
            vect_to_add[index] = np.count_nonzero(bins == index)
        self.pi_values = self.pi_values + vect_to_add
        self.ndata = self.ndata + len(data)
        self.pi_values = self.pi_values / self.ndata
        return

class Data_set_Handler:

    def __init__(self, dimensions_number):
        self.dimensions_number = dimensions_number
        self.gauss0 = ccm.random_gaussian(self.dimensions_number)
        return

    def return_equal_batch(self, B):
        return np.random.multivariate_normal(self.gauss0[0], self.gauss0[1], B)

    def generate_similar_batch(self, B, target_sKL):
        rot, shift = ccm.compute_roto_translation(self.gauss0, target_sKL)
        gauss1 = ccm.rotate_and_shift_gaussian(self.gauss0, rot, shift)
        return np.random.multivariate_normal(gauss1[0], gauss1[1], B)

def create_bins_combination(bins_number, minN):
    gauss = ccm.random_gaussian(bins_number)
    histogram = np.random.multivariate_normal(gauss[0], gauss[1], 1)
    histogram = histogram[0]
    histogram = np.abs(histogram)
    summa = np.sum(histogram)
    histogram = histogram / summa
    if minN == 0:
        return histogram
    histogram = np.sort(histogram)
    return histogram

