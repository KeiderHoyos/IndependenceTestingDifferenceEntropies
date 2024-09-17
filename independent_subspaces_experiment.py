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
parser.add_argument('-epochs', '--epochs', required = False, default = 100, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.085, type = float)
parser.add_argument('-batch_size', '--batch_size', required = False, default = None, type = int)
parser.add_argument('-grid_search_min', '--grid_search_min', required = False, default = 0, type = int)
parser.add_argument('-grid_search_max', '--grid_search_max', required = False, default = 1, type = int)

args = parser.parse_args()


def generate_ISA(n,d,sigma_normal,alpha, seed = 0):
    np.random.seed(seed)
    x = np.concatenate((np.random.normal(-1, sigma_normal, n//2), np.random.normal(1, sigma_normal, n//2)))
    y = np.concatenate((np.random.normal(-1, sigma_normal, n//2), np.random.normal(1, sigma_normal, n//2)))
    p = np.random.permutation(n)
    y_p = y[p]

    D = np.zeros([2,n])
    D[0,:] = x
    D[1,:] = y_p

    theta = np.pi/4*alpha
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    D_R = R@D
    X_mix = D_R[0,:].reshape(-1,1)
    Y_mix = D_R[1,:].reshape(-1,1)

    X_z = np.random.randn(n,d-1)
    Y_z = np.random.randn(n,d-1)

    X_con = np.concatenate((X_mix,X_z), axis=1)
    Y_con = np.concatenate((Y_mix,Y_z), axis=1)

    m_x = ortho_group.rvs(dim=d)
    m_y = ortho_group.rvs(dim=d)

    X = (m_x@X_con.T).T
    Y = (m_y@Y_con.T).T
    
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
    sigma_normal = 0.1 # 0.01 Check github code for the correct value (0.1 here: https://github.com/renyixin666/HSIC-LK/commit/a718bc42228ee82cbb3e7719ee7c9d6b8a1f62a2), according to paper 0.01
    n = 128
    d = 4
    alphas = np.linspace(0,1,10)
    n_alphas = len(alphas)
    n_tests = 11
    test_power = np.zeros([n_tests,n_alphas, test_num])
    

    for i, alpha in enumerate(alphas):
        for j in range(test_num):
            print('alpha:', alpha, 'repetition: ', j)
            X, Y =generate_ISA(n,d,sigma_normal,alpha, seed)
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

            # alpha = 0.125, Rényi Entropies    
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor,  alpha = 0.125,  type_bandwidth= 'isotropic',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[6, i, j] = float(results_dime['h0_rejected'])
        
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 0.125,  type_bandwidth= 'diagonal',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[7, i, j] = float(results_dime['h0_rejected'])

            # HSIC tests
            hsic0 = IndpTest_naive(X_tensor, Y_tensor, alpha=0.05, n_permutation=100, kernel_type="Gaussian", null_gamma = True)
            results_all0 = hsic0.perform_test()
            test_power[8, i, j] = float(results_all0['h0_rejected'])

            hsic1 = IndpTest_LKGaussian(X_tensor, Y_tensor, device, alpha=0.05, n_permutation=100, null_gamma = True, split_ratio = 0.5)
            results_all1 = hsic1.perform_test(debug = -1, if_grid_search = True)
            test_power[9, i, j] = float(results_all1['h0_rejected'])

            hsic2 = IndpTest_LKWeightGaussian(X_tensor, Y_tensor, device, alpha=0.05, n_permutation=100, null_gamma = True, split_ratio = 0.5)
            results_all2 = hsic2.perform_test(debug = -1, if_grid_search = True)
            test_power[10, i, j] = float(results_all2['h0_rejected'])


            if args.parallel:
                seed += repetitions
            else:
                seed += 1  
    np.savez(fname, *test_power)

if __name__ == "__main__":
    run()  