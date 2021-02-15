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

lamb = 0.1
bins_number = 8
statistic = qt.tv_statistic
alpha = [0.01]
nu = 32
desired_ARL0 = 100
training_rounds = 10
trainsition_rounds = 20
initial_pi_values = ext.create_bins_combination(bins_number, nu)
dimenions_number = 4

def plot_ARL0_once(max_lenght, desired_ARL0):
    learner = Paired_Learner(desired_ARL0= desired_ARL0, nu = nu, initial_pi_values = initial_pi_values, alpha=alpha,
                             statistic=statistic, lamb=lamb, training_rounds= training_rounds, transition_rounds=trainsition_rounds)
    handler = ext.Data_set_Handler(dimenions_number)
    batch = handler.return_equal_batch(10 * nu)
    stopped = learner.play_round(batch)
    for round in range(1, max_lenght):
        batch = handler.return_equal_batch(nu)
        stopped = learner.play_round(batch)
        if stopped:
            break
    return round

def plot_ARL0(exp_number, max_lenght):
    rounds = []
    for exp in range(exp_number):
        round = plot_ARL0_once(max_lenght, desired_ARL0)
        rounds.append(round)
    plt.boxplot(rounds)
    plt.title('ARL0')
    plt.show()

plot_ARL0(10, 1000)