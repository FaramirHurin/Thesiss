import libquanttree as qt

#We can train nets here, store in a file and use them during code

data_Dimension = 3
nu = 32
statistic = qt.tv_statistic

"""
frame = pd.read_csv('Asymptotic_0_5')
alpha = [0.5]
tested = 100
values = np.zeros([tested, 3])
for index in range(tested):
    histogram = frame.iloc[index, 1:-1]
    threshold_computer = aux.Alternative_threshold_computation(histogram, nu, statistic)
    threshold = threshold_computer.compute_threshold(alpha, 3000)
    threshold2 = threshold_computer.compute_threshold(alpha, 3000)
    threshold_stored = frame.iloc[index, -1]
    values[index] = np.array([threshold_stored, threshold, threshold2])
final_frame = pd.DataFrame(values)
final_frame['differences with stored'] = np.abs(values[:,0] - values[:,1])
final_frame['inner differences'] = np.abs(values[:,1] - values[:,2])
final_frame =  final_frame.rename(columns={0:'stored', 1:'predictions1', 2: 'predictions2'})
print('alpha = '+str(alpha))
print(final_frame.describe())
"""
"""
Test N and thr 0.5
frame = pd.read_csv('File_N_and_thr_0_5')
alpha = [0.5]
tested = 100
values = np.zeros([tested, 3])
for index in range(tested):
    histogram = frame.iloc[index, 1:-2]
    N =     threshold_stored = frame.iloc[index, -1]
    tree = qt.QuantTree(histogram)
    tree.ndata = int(N)
    threshold_computer = qt.ChangeDetectionTest(tree, nu, statistic)
    threshold = threshold_computer.estimate_quanttree_threshold(alpha, 3000)
    threshold2 = threshold_computer.estimate_quanttree_threshold(alpha, 3000)
    threshold_stored = frame.iloc[index, -2]
    values[index] = np.array([threshold_stored, threshold, threshold2])
final_frame = pd.DataFrame(values)
final_frame['differences with stored'] = np.abs(values[:,0] - values[:,1])
final_frame['inner differences'] = np.abs(values[:,1] - values[:,2])
final_frame =  final_frame.rename(columns={0:'stored', 1:'predictions1', 2: 'predictions2'})
print('alpha = '+str(alpha))
print(final_frame.describe())
"""
"""
# Test Asymptotic 0.1
regressor = MLPRegressor(300, early_stopping=True, validation_fraction=0.05)
frame = pd.read_csv('Asymptotic_0._1')
alpha = [0.01]
histograms = frame.iloc[:, 1:-1]
thresholds = frame.iloc[:, -1]
regressor.fit(histograms, thresholds)
print(regressor.best_validation_score_)
tested = 40
values = np.zeros([tested, 3])
for index in range(tested):
    histogram = aux.create_bins_combination(8, 100)
    threshold_computer = aux.Alternative_threshold_computation(histogram, nu, statistic)
    threshold = threshold_computer.compute_threshold(alpha, 10000)
    threshold2 = threshold_computer.compute_threshold(alpha, 10000)
    histogram = np.array(histogram)
    threshold3 = regressor.predict(histogram.reshape(1, -1))
    values[index] = np.array([ threshold, threshold2, threshold3])
final_frame = pd.DataFrame(values)
final_frame['inner differences'] = np.abs(values[:,0] - values[:,1])
final_frame['differences'] = np.abs(values[:,0] - values[:,2])
final_frame =  final_frame.rename(columns={0:'trhesholds', 1 : 'thresholds2', 2:'regressions' })
print('alpha = '+str(alpha))
print('B = 10000')
print(final_frame.head(8))
print(final_frame.describe())
"""
"""
#Test file with N and thr 0.01
regressor = MLPRegressor
(50, early_stopping=True, learning_rate='invscaling', solver = 'adam', validation_fraction= 0.1, verbose=True, alpha=0.1, max_iter=10000, n_iter_no_change=40)
frame = pd.read_csv('File_N_and_thr_0_01')
alpha = [0.01]
histograms = frame.iloc[:, 1:-2]
thresholds = frame.iloc[:, -2]
N = frame.iloc[:, -1]
histograms['N'] = N
#scaler = StandardScaler()
#scaler.fit_transform(histograms)
regressor.fit(histograms, thresholds)
print(regressor.best_validation_score_)
tested = 40
values = np.zeros([tested, 3])
for index in range(tested):
    histogram = aux.create_bins_combination(8, 100)
    tree = qt.QuantTree(histogram)
    number = np.random.randint(10, 1000)
    tree.ndata =number
    threshold_computer = qt.ChangeDetectionTest(tree, nu, statistic)
    threshold = threshold_computer.estimate_quanttree_threshold(alpha, 3000)
    threshold2 = threshold_computer.estimate_quanttree_threshold(alpha, 3000)
    rich_histogram = np.append(histogram, float(number))
    #rich_histogram = scaler.transform(rich_histogram.reshape(1, -1))
    threshold3 = regressor.predict(rich_histogram.reshape(1, -1))
    values[index] = np.array([ threshold, threshold2, threshold3])
final_frame = pd.DataFrame(values)
final_frame['inner differences'] = np.abs(values[:,0] - values[:,1])
final_frame['differences'] = np.abs(values[:,0] - values[:,2])
final_frame =  final_frame.rename(columns={0:'trhesholds', 1 : 'thresholds2', 2:'regressions' })
print('alpha = '+str(alpha))
print('B = 10000')
print(final_frame.head(8))
print(final_frame.describe())
print ('r2 is ' + str(r2_score(final_frame['trhesholds'], final_frame['regressions'])))
print('inner r2 is ' + str(r2_score(final_frame['trhesholds'], final_frame['thresholds2'])))
"""