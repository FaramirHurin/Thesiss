import numpy as np
import libccm as ccm

dim = 8
ndata = 4096
# generate a random multivariate Gaussian distribution phi_0
gauss0 = ccm.random_gaussian(dim)

# generate a stationary dataset from phi_0
data = np.random.multivariate_normal(gauss0[0], gauss0[1], ndata)

target_sKL = 1
nbatch = 32
rot, shift = ccm.compute_roto_translation(gauss0, target_sKL)
# compute the alternative distribution phi_1
gauss1 = ccm.rotate_and_shift_gaussian(gauss0, rot, shift)
# generate batch from phi_1
batch = np.random.multivariate_normal(gauss1[0], gauss1[1], nbatch)
