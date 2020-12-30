import libquanttree as qt
import numpy as np
import matplotlib.pyplot as plt
import quant_garden as gard
import auxilium as aux
import libccm as ccm
import auxilium_test as aux_test

#WE ARE TESTING WITH NU = 32

bins_number = 16 #8
dimension_number = 6
nu = 16
statistic = qt.tv_statistic
alpha = [0.01]
K = 30 #Number of trees to keep in the garden
beta = [0.5]
min_N = aux_test.min_N
max_N = aux_test.max_N
initial_pi_values = np.ones(bins_number)/bins_number
gard.bins_number = bins_number
gard.initial_pi_values = np.ones(bins_number)/bins_number


def test_qt_ARLO0():
    handler = aux.Data_set_Handler(dimension_number)
    false_positive = 0
    stop_time = 0
    tree = qt.QuantTree(initial_pi_values)
    training_set = handler.return_equal_batch(10 * nu)
    tree.build_histogram(training_set)
    threshold = qt.ChangeDetectionTest(tree, nu, statistic).estimate_quanttree_threshold(alpha, 20000)
    while false_positive == 0:
        batch = handler.return_equal_batch(nu)
        stat = statistic(tree, batch)
        if stat > threshold:
            false_positive = 1
        stop_time += 1
        if stop_time > 600:
            break
    print('stop time classic QT' + str(stop_time))
    return stop_time

def test_ARL0(check_garden):
    handler = aux.Data_set_Handler(dimension_number)
    false_positive = 0
    stop_time = 0
    garden = gard.Quant_garden(K,statistic, nu, alpha, beta, bins_number, min_N, max_N, check_garden)
    while false_positive == 0:
        batch = handler.return_equal_batch(nu)
        false_positive = garden.play_round(batch)
        stop_time += 1
        if stop_time > 6000:
            break
    print('stop time ' + str(stop_time))
    return stop_time


def test_ARL1(change_time, SKL,check_garden):
    handler = aux.Data_set_Handler(dimension_number)
    positive = 0
    time = 0
    garden = gard.Quant_garden(K, statistic, nu, alpha, beta, bins_number, min_N, max_N, check_garden)
    while positive == 0:
        if time < change_time:
            batch = handler.return_equal_batch(nu)
            false_positive = garden.play_round(batch)
        else:
            batch = handler.generate_similar_batch(nu, SKL)
            positive = garden.play_round(batch)
        time += 1
        if time > 6000:
            break
    if time < change_time:
        print ('False positive')
    print('stop time ' + str(time - change_time))
    if time > change_time:
        return time - change_time
    else:
        return 100 #penalty

def test_QT_ARL1(change_time, SKL):
    handler = aux.Data_set_Handler(dimension_number)
    positive = 0
    time = 0
    tree = qt.QuantTree(initial_pi_values)
    training_set = handler.return_equal_batch(5 * nu)
    tree.build_histogram(training_set)
    threshold = qt.ChangeDetectionTest(tree, nu, statistic).estimate_quanttree_threshold(alpha, 20000)
    while positive == 0:
        if time < change_time:
            batch = handler.return_equal_batch(nu)
            stat = statistic(tree, batch)
            false_positive = stat > threshold
        else:
            batch = handler.generate_similar_batch(nu, SKL)
            stat = statistic(tree, batch)
            positive = stat > threshold
        time += 1
        if time > 6000:
            break
    if time < change_time:
        print ('False positive QT')
    print('stop time QT' + str(time - change_time))
    if time > change_time:
        return time - change_time
    else:
        return 100

def plot_ARL0(points_to_plot, check_garden):
    print('Start')
    garden_points = []
    normal_points = []
    for index in range(points_to_plot):
        garden_points.append(test_ARL0(check_garden))
        normal_points.append(test_qt_ARLO0())
    fig, axs = plt.subplots(2)
    axs[0].boxplot(garden_points, notch=False, showfliers  = False)
    axs[0].set_title('Garden ARLO-tv')
    axs[0].legend('We were checking: ' + str(check_garden))
    axs[1].boxplot(normal_points, notch=False, showfliers  = False)
    axs[1].set_title('Normal ARL0-tv')
    plt.show()

    print (' Normal mean-t' +str(np.mean(normal_points)))
    print ('Garden mean-t' + str(np.mean(garden_points)))
    print ('Normal median - t ' + str(np.median(normal_points)))
    print ('Garden median - t ' + str(np.median(garden_points)))

    return

def plot_ARL1(points_to_plot, change_time, SKL):
    print('Start')
    garden_points = []
    normal_points = []
    garden_points_with_check = []
    for index in range(points_to_plot):
        garden_points.append(test_ARL1(change_time, SKL, False))
        normal_points.append(test_QT_ARL1(change_time, SKL))
        garden_points_with_check.append(test_ARL1(change_time, SKL, True))
        array = np.array([normal_points, garden_points, garden_points_with_check])
    plt.boxplot(array.T, labels = ['normal FP1', ' garden FP1', 'full garden FP1'])
    plt.title('FP1')
    plt.show()

    print (' Normal mean FP1-t' +str(np.mean(normal_points)))
    print ('Garden mean FP1-t' + str(np.mean(garden_points)))
    print ('Normal median FP1 - t ' + str(np.median(normal_points)))
    print ('Garden median FP1 - t' + str(np.median(garden_points)))

    return

#print(str(statistic.__name__) + ': expected = ' + str(1/alpha[0]))
plot_ARL1(40, 20, 0.3)
print ('sk = 1')