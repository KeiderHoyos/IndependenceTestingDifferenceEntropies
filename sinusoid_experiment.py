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
parser.add_argument('-dime_perm', '--dime_perm', required = False, default = 10, type = int)
parser.add_argument('-epochs', '--epochs', required = False, default = 250, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.01, type = float)
parser.add_argument('-batch_size', '--batch_size', required = False, default = None, type = int)
parser.add_argument('-grid_search_min', '--grid_search_min', required = False, default = -1, type = int)
parser.add_argument('-grid_search_max', '--grid_search_max', required = False, default = 4, type = int)

args = parser.parse_args()


def Sinusoid(x, y, w):
    return 1 + np.sin(w*x)*np.sin(w*y)

def Sinusoid_Generator(n,w, seed = 0):
    np.random.seed(seed)
    i = 0
    output = np.zeros([n,2])
    while i < n:
        U = np.random.rand(1)
        V = np.random.rand(2)
        x0 = -np.pi + V[0]*2*np.pi
        x1 = -np.pi + V[1]*2*np.pi
        if U < 1/2 * Sinusoid(x0,x1,w):
            output[i, 0] = x0
            output[i, 1] = x1
            i = i + 1
    X = output[:,0:1]
    Y = output[:,1:]
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
    w = 3

    sample_sizes = (300, 400,500,600)
    n_samples = len(sample_sizes)
    n_tests = 9
    test_power = np.zeros([n_tests,n_samples, test_num])
    

    for i, n in enumerate(sample_sizes):
        for j in range(test_num):
            print('sample size:', n, 'repetition: ', j)
            X, Y = Sinusoid_Generator(n, w, seed)
            Y = Y.reshape(-1,1)
            X_tensor, Y_tensor = torch.tensor(X, device=device), torch.tensor(Y,device=device)

            # alpha = 1.0, von Neumann Entropies
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 1.0, type_bandwidth= 'isotropic',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[0, i, j] = float(results_dime['h0_rejected'])
            
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 1.0,  type_bandwidth= 'diagonal',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[1, i, j] = float(results_dime['h0_rejected'])

            # alpha = 2.0, Rényi Entropies
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 2.0, type_bandwidth= 'isotropic',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[2, i, j] = float(results_dime['h0_rejected'])
            
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 2.0, type_bandwidth= 'diagonal', 
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[3, i, j] = float(results_dime['h0_rejected'])

            # alpha = 0.5, Rényi Entropies    
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor,  alpha = 0.5, type_bandwidth= 'isotropic',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[4, i, j] = float(results_dime['h0_rejected'])
        
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 0.5, type_bandwidth= 'diagonal',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
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