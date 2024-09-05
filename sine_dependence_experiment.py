import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats as stats
from scipy.stats import gamma
from scipy.stats import ortho_group

# import cupy as cp
import cupyx.scipy 

from Test.hsic_naive import IndpTest_naive
from Test.hsic_lkgau import IndpTest_LKGaussian
from Test.hsic_lklap import IndpTest_LKLaplace
from Test.hsic_lkwgau import IndpTest_LKWeightGaussian
from Test.hsic_lkwlap import IndpTest_LKWeightLaplace
from Test.hsic_lkselect import IndpTest_LKSelect_GauLap
from Test.hsic_kselect import IndpTest_KSelect
from Test.dime_gaussian import IndpTest_DIME
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)
parser.add_argument('-rId', '--repId',required=False, default= 1, help = 'repetition Id to store the results', type=int)
parser.add_argument('-datafolder', '--DATAFOLDER', required = True, type = str )
parser.add_argument('-parallel', '--parallel', required = False, default = False, type = bool)

args = parser.parse_args()


def sinedependence(n,d,seed = 0):
    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, n)
    Z = np.random.randn(n)
    Y = 20*np.sin(4*np.pi*(X[:,0]**2 + X[:,1]**2))+Z 
    return X,Y

def run():
    repetitions = 100
    fname = args.DATAFOLDER + '/' + str(args.experiment_name) + str(args.repId) + '.npz'
    if args.parallel:
        test_num = 1
        seed = 0 + args.repId - 1 
    else:
        test_num = repetitions
        seed = 0 
    device = torch.device('cuda')
    d = 3

    sample_sizes = (300, 600,900,1200)
    n_samples = len(sample_sizes)
    n_tests = 9
    test_power = np.zeros([n_tests,n_samples, test_num])
    

    for i, n in enumerate(sample_sizes):
        for j in range(test_num):
            print('sample size:', n, 'repetition: ', j)
            X, Y = sinedependence(n, d, seed)
            Y = Y.reshape(-1,1)
            X_tensor, Y_tensor = torch.tensor(X, device=device), torch.tensor(Y,device=device)

            # alpha = 1.0, von Neumann Entropies
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, dime_perm = 10, alpha = 1.0, isotropic = True)
            results_dime = dime_estimator.perform_test()
            test_power[0, i, j] = float(results_dime['h0_rejected'])
            
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, dime_perm = 10, alpha = 1.0, isotropic = False)
            results_dime = dime_estimator.perform_test()
            test_power[1, i, j] = float(results_dime['h0_rejected'])

            # alpha = 2.0, Rényi Entropies
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, dime_perm = 10, alpha = 2.0, isotropic = True)
            results_dime = dime_estimator.perform_test()
            test_power[2, i, j] = float(results_dime['h0_rejected'])
            
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, dime_perm = 10, alpha = 2.0, isotropic = False)
            results_dime = dime_estimator.perform_test()
            test_power[3, i, j] = float(results_dime['h0_rejected'])

            # alpha = 0.5, Rényi Entropies    
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, dime_perm = 10, alpha = 0.5, isotropic = True)
            results_dime = dime_estimator.perform_test()
            test_power[4, i, j] = float(results_dime['h0_rejected'])
        
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, dime_perm = 10, alpha = 0.5, isotropic = False)
            results_dime = dime_estimator.perform_test()
            test_power[5, i, j] = float(results_dime['h0_rejected'])

            # HSIC tests
            hsic0 = IndpTest_naive(X_tensor, Y_tensor, alpha=0.05, n_permutation=100, kernel_type="Gaussian", null_gamma = True)
            results_all0 = hsic0.perform_test()
            test_power[6, i, j] = float(results_all0['h0_rejected'])

            hsic1 = IndpTest_LKGaussian(X_tensor, Y_tensor, device, alpha=0.05, n_permutation=100, null_gamma = True, split_ratio = 0.5)
            results_all1 = hsic1.perform_test(debug = -1, if_grid_search = True)
            test_power[7, i, j] = float(results_all1['h0_rejected'])

            hsic2 = IndpTest_LKWeightGaussian(X_tensor, Y_tensor, device, alpha=0.05, n_permutation=100, null_gamma = True, split_ratio = 0.5)
            results_all2 = hsic2.perform_test(debug = -1, if_grid_search = True)
            test_power[8, i, j] = float(results_all2['h0_rejected'])

            if args.parallel:
                seed += repetitions
            else:
                seed += 1  
    np.savez(fname, *test_power)

if __name__ == "__main__":
    run()  