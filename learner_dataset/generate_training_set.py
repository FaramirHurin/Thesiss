import pandas as pd
import qtLibrary.libquanttree as qt

statistic = qt.tv_statistic
nu = 64
alpha = [0.01]
B = 4000

#Actual level = 4000

file = 'net_training_set/Thresholds_TV_1.csv'

frame = pd.read_csv('net_training_set/Thresholds_TV_1.csv')
frame.drop(frame.columns[0], axis = 1, inplace = True)
for index in range(2400, 3000):
    tree = qt.QuantTree(frame.iloc[index, :16])
    tree.ndata = int(frame.loc[index, 'N'])
    tree.ndata
    frame.loc[index, 'Thresholds'] = qt.ChangeDetectionTest(tree, nu, statistic).estimate_quanttree_threshold(alpha, B)
    current_value = index

frame.to_csv(file)