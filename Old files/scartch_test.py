from scratch import *
import pandas

def test_FP0():
    number_of_experiments = 40
    experiment_values_modified = np.zeros(number_of_experiments)
    experiment_values_original = np.zeros(number_of_experiments)
    experiment_values_regressor = np.zeros(number_of_experiments)


    datasets_number = 300
    for counter in range(number_of_experiments):

        #Creazione datasets
        dataSetCreator = DataSet_Creator()
        dataSet1 = dataSetCreator.createDataSet(bins_number)
        dataSet2 = dataSetCreator.createDataSet(datasets_number - bins_number)
        dataSet = np.append(dataSet1, dataSet2)

        #Creazione alberi
        histogramCreator = treeCreator(bins_number, initial_pi_values)
        tree1 = histogramCreator.createHistogram(dataSet1)
        tree2 = histogramCreator.createHistogram(dataSet)

        # Modifica histogram 1
        piModifier = PiModifier(tree1, len(dataSet1))
        tree1 = piModifier.modify_probabilities(dataSet2)
        batchSize = 40

        #Calcolo thresholds
        threshold_regressor = Regressor('pearson')
        threshold_procedure = Thresholds_numerical_procedure(alpha, 'pearson', bins_number)
        threshold_regressed1 = threshold_regressor.computeThreshold(tree1.pi_values)
        threshold1 = threshold_procedure.computeThreshold(tree1.pi_values)
        threshold2 = threshold_procedure.computeThreshold(tree2.pi_values)

        tester1 = Tester('pearson', tree1, threshold1)
        tester2 = Tester('pearson', tree2, threshold2)
        tester_regressor1 = Tester('pearson', tree1, threshold_regressed1)

        number_of_datasets_to_average = 200

        for counter in range(number_of_experiments):

            for inner_counter in range(number_of_datasets_to_average):

                batch = dataSetCreator.create_similar_batch(target_sKL, batchSize)

                if tester1.take_Decision(batch):
                    experiment_values_modified[counter] += 1
                if tester2.take_Decision(batch):
                    experiment_values_original[counter] += 1
                if tester_regressor1.take_Decision(batch):
                    experiment_values_regressor[counter] += 1

            experiment_values_modified[counter] = \
                experiment_values_modified[counter]/number_of_datasets_to_average

            experiment_values_original[counter] = \
                experiment_values_original[counter] / number_of_datasets_to_average

            experiment_values_regressor[counter] = \
                experiment_values_regressor[counter]/number_of_datasets_to_average


    fig, axs = plt.subplots(3)
    axs[0].boxplot(experiment_values_modified, notch=False,  showfliers=False)
    #plt.title('experiment_values_modified')
    axs[1].boxplot(experiment_values_original, notch=False,  showfliers=False)

    axs[2].boxplot(experiment_values_regressor, notch=False,  showfliers=False)
    #plt.title('experiment_values_original')
    plt.show()


def test_type1_error():

    #Creazione dei dataset
    dataset_number = 10000
    dataSetCreator = DataSet_Creator()
    dataSet1 = dataSetCreator.createDataSet(datasets_number)
    dataSet2 = dataSetCreator.createDataSet(datasets_number)
    dataSet = np.append(dataSet1, dataSet2)


    #Creazione histograms
    histogramCreator = DataSet_Creator(bins_number, initial_pi_values)
    tree1 = histogramCreator.createHistogram(dataSet1)
    tree2 = histogramCreator.createHistogram(dataSet)


    #Modifica histogram 1
    piModifier = PiModifier(tree1, len(dataSet1))
    tree1 = piModifier.modify_probabilities(dataSet2)



    number_of_experiments_to_average = 5000
    number_of_recognized = [0,0,0]
    batchSize = 1000

    # Threshold computation
    #TODO Add regressor variant
    threshold_regressor = Regressor('pearson')
    threshold_procedure = Thresholds_numerical_procedure(alpha, 'pearson', bins_number)
    threshold1 = threshold_procedure.computeThreshold(tree1)
    threshold2 = threshold_procedure.computeThreshold(tree2)
    threshold_regressed1 = threshold_regressor.computeThreshold(tree1.pi_values)

    for counter in range(number_of_experiments_to_average):

        batch = dataSetCreator.createDataSet(batchSize)

        tester1 = Tester('pearson', tree1, threshold1)
        tester2 = Tester('pearson', tree2, threshold2)
        tester_regressor1 = Tester('pearson', tree1, threshold_regressed1)

        if tester1.take_Decision(batch):
            number_of_recognized [0] += 1
        #Test if batch it is recognized by the QT1 and QT2, if so cond = true
        if tester2.take_Decision(batch):
            number_of_recognized[1] += 1
        if tester_regressor1.take_Decision(batch):
            number_of_recognized[2] += 1

    return number_of_recognized[0]/number_of_experiments_to_average, number_of_recognized[1]/number_of_experiments_to_average


def test_accuracy_and_precision():

    number_of_experiments_to_average = 50

    #Creazione dei dataset
    dataset_number = 300
    dataSetCreator = DataSet_Creator()
    dataSet1 = dataSetCreator.createDataSet(6)
    dataSet2 = dataSetCreator.createDataSet(datasets_number)
    dataSet = np.append(dataSet1, dataSet2)

    # Creazione histogram 1
    histogramCreator = treeCreator(bins_number, initial_pi_values)
    tree1 = histogramCreator.createHistogram(dataSet1)
    tree2 = histogramCreator.createHistogram(dataSet)

    # Modifica histogram 1
    piModifier = PiModifier(tree1, len(dataSet1))
    tree1 = piModifier.modify_probabilities(dataSet2)



    batchSize = 30

    all_sKL = np.linspace(0, 10, 50)
    # Threshold computation
    threshold_procedure = Thresholds_numerical_procedure(alpha, 'pearson', bins_number)
    threshold_regressor = Regressor('pearson')

    threshold1 = threshold_procedure.computeThreshold(tree1.pi_values)
    threshold2 = threshold_procedure.computeThreshold(tree2.pi_values)
    threshold_regressed1 = threshold_regressor.computeThreshold(tree1.pi_values)

    accuracy0 = []
    accuracy1 = []
    accuracy2 = []
    precision0 = []
    precision1 = []
    precision2 = []

    number_of_recognized = np.zeros((50, 3))
    target_counter = 0
    for target_sKL in all_sKL:

        tester1 = Tester('pearson', tree1, threshold1)
        tester2 = Tester('pearson', tree2, threshold2)
        tester_regressor = Tester('pearson', tree1, threshold_regressed1)

        for counter in range(number_of_experiments_to_average):

            batch = dataSetCreator.create_similar_batch(target_sKL, batchSize)

            if tester1.take_Decision(batch):
                number_of_recognized[target_counter][0] += 1
            # Test if batch it is recognized by the QT1 and QT2, if so cond = true
            if tester2.take_Decision(batch):
                number_of_recognized[target_counter][1] += 1
             #   print (tester2.statistic_value)
            if tester_regressor.take_Decision(batch):
                number_of_recognized[target_counter][2] += 1


        if target_counter == 0:
            accuracy0.append(number_of_recognized[0][0] / number_of_experiments_to_average)

            accuracy1.append(number_of_recognized[0][1]/ number_of_experiments_to_average )

            accuracy2.append(number_of_recognized[0][2] / number_of_experiments_to_average)

            precision0.append(1)

            precision1.append(1)

            precision2.append(1)
        else:
            accuracy0.append((number_of_recognized[0][0] + number_of_experiments_to_average -
                            number_of_recognized[target_counter][0])/number_of_experiments_to_average/2)

            accuracy1.append ((number_of_recognized[0][1] + number_of_experiments_to_average -
                            number_of_recognized[target_counter][1])/number_of_experiments_to_average/2)

            accuracy2.append((number_of_recognized[0][2] + number_of_experiments_to_average -
                            number_of_recognized[target_counter][2])/number_of_experiments_to_average/2)

            precision0.append(number_of_recognized[0][0]
                            /(number_of_recognized[0][0] + number_of_recognized[target_counter][0]))

            precision1.append(number_of_recognized[0][1]
                            /(number_of_recognized[0][1] + number_of_recognized[target_counter][1]))

            precision2.append(number_of_recognized[0][2]
                              / (number_of_recognized[0][2] + number_of_recognized[target_counter][2]))

        target_counter = target_counter + 1

    #plt.plot(all_sKL, accuracy0, label = 'Accuracy: Modified tree')
    #plt.plot(all_sKL, accuracy1, label='Accuracy: Original tree')
    plt.plot(all_sKL, precision0, label='Precision: Modified tree')
    plt.plot(all_sKL, precision1, label='Precision: Original tree')
    #plt.plot(all_sKL, accuracy2, label = 'Accuracy: Modified tree with regressor')
    plt.plot(all_sKL, precision2, label='Precision: Original tree with regressor')


    plt.legend()
    plt.show()

    return

#Stores dataset of histograms to train the regressor
def store_data_set(statistic_used):
    thresholds = []
    histograms = DataSet_Creator.generate_Random_histograms()
    for histogram in histograms:
        histogram.sort()
        numerical_procedure = Thresholds_numerical_procedure(
            alpha, statistic_used, bins_number)
        thresh = numerical_procedure.computeThreshold(histogram)
        thresholds.append(thresh)
    thresholds = np.array(thresholds)
    histograms = np.insert(histograms, 0, thresholds, axis = 1)
    df = pandas.DataFrame(histograms)
    df.to_csv("DataBase2")

#Boxplot of the histograms dataset
def plot_thresholds():
    dataSet = DataSet_Creator.load_data_set()
    dataSet = dataSet.to_numpy()
    thresholds = dataSet[:,1]
    plt.boxplot(thresholds, notch=False, whis =(0, 95), showfliers=False)
    plt.title('Threshold values, 5% removed')
    plt.show()

#Double check: outliers don't come from variance in the generation process
def studyThresholds():
    dataSet = DataSet_Creator.load_data_set()
    dataSet = dataSet.to_numpy()
    for elem in dataSet:
        if elem[1] > np.percentile(dataSet[:,1], 95):
            print (elem[1], Thresholds_numerical_procedure(alpha, 'pearson',bins_number).computeThreshold(elem[2:]))

#Tells if outiler value -> smaller minimal value
def analyzeDataSet():
    dataSet = DataSet_Creator.load_data_set()
    dataSet = dataSet.to_numpy()
    vars = np.zeros(len(dataSet))
    mins = np.zeros(len(dataSet))
    counter = 0
    percentile = 95
    for elem in dataSet:
        vars[counter] = np.var(elem[2:5])
        mins[counter] = np.min(elem[2:5])
        counter = counter + 1
    for elem in dataSet:
        if elem[1] > np.percentile(dataSet[:,1], percentile):
            if  np.min(elem[2:5]) > np.percentile(mins, (100 -  percentile)*1.5):
                print ("Minimal problem")
                print(np.min(elem[2:5]), np.percentile(mins, (100 - percentile)*2))
                return False
            """
            if  np.var(elem[2:5]) < np.percentile(vars, 98):
                print ("variance problem")
                print(np.var(elem[2:5]), np.percentile(vars, 98))
                return False
            """
            #print(elem[1], np.var(elem[2:5]), np.min(elem[2:5]))
            #If not bigger than meean var
    #print (np.mean(vars), np.mean(mins))
    return True


#Double check in order to assure that outliers bring to bad performance
def test_particular_histogram(histogram):
    batchSize = 500
    dataset_number = 2000
    dataSetCreator = DataSet_Creator()
    dataSet = dataSetCreator.createDataSet(datasets_number)
    histogramCreator = treeCreator(bins_number, histogram)
    tree = histogramCreator.createHistogram(dataSet)
    all_sKL = np.linspace(0, 10, 100)
    # Threshold computation
    threshold_procedure = Thresholds_numerical_procedure(alpha, 'pearson', bins_number)
    threshold = threshold_procedure.computeThreshold(tree.pi_values)
    number_of_experiments_to_average = 200

    accuracy = []
    precision = []

    number_of_recognized = np.zeros(number_of_experiments_to_average)
    target_counter = 0

    for target_sKL in all_sKL:

        tester1 = Tester('pearson', tree, threshold)

        for counter in range(number_of_experiments_to_average):

            batch = dataSetCreator.create_similar_batch(target_sKL, batchSize)

            if tester1.take_Decision(batch):
                number_of_recognized[target_counter] += 1
            # Test if batch it is recognized by the QT1 and QT2, if so cond = true

             #   print (tester2.statistic_value)
        if target_counter == 0:
            accuracy.append(number_of_recognized[0] / number_of_experiments_to_average)


            precision.append(1)

        else:
            accuracy.append((number_of_recognized[0] + number_of_experiments_to_average -
                            number_of_recognized[target_counter])/number_of_experiments_to_average/2)


            precision.append(number_of_recognized[0]
                            /(number_of_recognized[0] + number_of_recognized[target_counter]))


        target_counter = target_counter + 1

    plt.plot(all_sKL, accuracy, label = 'Accuracy: Tree with given histogram')
    plt.plot(all_sKL, precision, label='Precision: Tree with given histogram')

    plt.legend()
    plt.show()

    return

def test_regressor():

    regressor = Regressor('pearson')

    histogram = np.random.rand(4)
    histogram = histogram / sum(histogram)

    numerical_procedure = Thresholds_numerical_procedure(alpha, 'pearson', bins_number)
    original = numerical_procedure.computeThreshold(histogram)

    regressor.tune_Model()

    estimate = regressor.computeThreshold(histogram)
    print(original, estimate)

#test_accuracy_and_precision()
#store_data_set('pearson')
#plot_thresholds()
#studyThresholds()
#hist = [0.0003337097421917848,0.07790243972968347,0.26157786184878684,0.6601859886793379]
#test_particular_histogram(hist)
#print(analyzeDataSet())
#test_FP0()
#test_regressor()