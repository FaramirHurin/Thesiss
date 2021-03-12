import pandas as pd
import qtLibrary.libquanttree as qt
import numpy as np
import main_code.auxiliary_project_functions as aux
import pickle as pk

statistic = qt.tv_statistic
nu = 128
alpha = [0.01]
B = 20000
bins = 32
minN = 100
maxN = 10000
#Actual level = 4000

file_name = 'final_thresholds.csv'

legend = {
    'nu': nu,
    'bins': 32,
    'minN': minN,
    'maxN': maxN,
    'statistic': statistic.__name__,
    'B': B,
    'alpha': alpha,
    'file Name': file_name
}



table = np.zeros([100000, bins + 2])
print(table.shape)
for index in range(table.shape[0]):
    table[index,:bins] = aux.create_bins_combination(bins)
    table[index, -2] = np.random.randint(low = minN, high = maxN)
    tree = qt.QuantTree( table[index,:bins])
    tree.ndata = int(table[index, -2])
    table[index, -1] = qt.ChangeDetectionTest(tree, nu, statistic).estimate_quanttree_threshold(alpha, B)

frame = pd.DataFrame(table)
frame.to_csv(file_name)
with open('final_legend.pickle', 'wb') as fil:
    pk.dump(legend, fil)

print(legend)

