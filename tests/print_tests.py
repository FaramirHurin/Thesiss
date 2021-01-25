#We let plots out of the functions

#EXTENDEND

#Inputs: phi0, N, percentage, nu, binsNumber, dataDimension
#Outputs: QT and updating-QT FPR over multiple runs
def test_modified_FPR(phi0, nu, bins_number, data_dimension, N, percentage):
    experiments_number = 0

    qt_FPR_over_runs = []
    new_FPR_over_runs = []
    return qt_FPR_over_runs, new_FPR_over_runs

#Inputs: phi0, phi1, N, percentage, nu, binsNumber, dataDimension
#Outputs QT and updating-QT powers box plots
def test_modified_power():
    qt_power_over_runs = []
    new_power_over_runs = []
    return qt_power_over_runs, new_power_over_runs

#Inputs: phi0, N, percentage, nu, binsNumber, dataDimension
#Outputs: QT and asymptotic updating-QT FPR box plots
def test_asymptotic_FPR():
    qt_FPR_over_runs = []
    asymptotic_FPR_over_runs = []
    return qt_FPR_over_runs, asymptotic_FPR_over_runs

#Inputs: phi0, N, percentage, nu, binsNumber, dataDimension, NN hyperparameters(?)
#Outputs: QT and updating-QT FPR box plots

def test_NN_FPR():
    qt_FPR_over_runs = []
    regressed_FPR_over_runs = []
    return qt_FPR_over_runs, regressed_FPR_over_runs

#Dynamic QuantTree

#Inputs: phi0, N, percentage, nu, binsNumber, dataDimension, NN hyperparameters, rounds, lambda
#Outputs: Box plot of the average of the statistic given multiple tun
def check_stat_average():
    return

#It basically runs check_stat_average() with different distributions and verifies they have the same box plots
def check_statistic_independence():
    return

#Tries to tell wether the ARL0 can be controlled by controlling Beta (if previous is true, we can use a uniform)
def verify_beta_ARL0_correlation():
    return

#Garden tests

def compare_FP0_with_gaussian_mixture():
    return

def compare_power_with_gaussian_mixture():
    return

def compare_real_FP0_with_gaussian_mixture():
    return

def compare_real_power_with_gaussian_mixture():
    return