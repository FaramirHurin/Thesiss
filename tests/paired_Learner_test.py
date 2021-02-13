from paired_learner import  Paired_Learner
import extendedQuantTree as ext
import matplotlib.pyplot as plt
import qtLibrary.libquanttree as qt

"""
Generates multiple batches and feeds them to the algorithm.
We want to show that:
        2) Some sort of control on reaching phase 3 (FPR phase 2)
        3) Some sort of control on power during phase 2
        4) Control on both power and FPR during phase 3, when the two learner work together
"""

initial_pi_values = ext.create_bins_combination()
lamb = 0
statistic = qt.tv_statistic
alpha = 0
nu = 0
desired_ARL0 = 0
training_rounds = 0
trainsition_rounds = 0

def plot_ARL0_once(N):
    learner = Paired_Learner()
    handler = ext.Data_set_Handler()
    for round in range(N):
        batch = handler.return_equal_batch()
        stopped = learner.play_round()
        if stopped:
            break
    return round

def plot_ARL0(exp_number, N):
    rounds = []
    for exp in exp_number:
        round = plot_ARL0_once(N)
        rounds.append(round)
    plt.boxplot(rounds)
    plt.title('ARL0')
    plt.show()