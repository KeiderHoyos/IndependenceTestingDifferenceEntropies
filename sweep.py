import numpy as np
import numpy as np
import time
import itertools
import torch
from scipy.stats import gamma
from scipy.stats import ortho_group
from Test.hsic_naive import IndpTest_naive
from Test.hsic_lkgau import IndpTest_LKGaussian
from Test.hsic_lklap import IndpTest_LKLaplace
from Test.hsic_lkwgau import IndpTest_LKWeightGaussian
from Test.hsic_lkwlap import IndpTest_LKWeightLaplace
from Test.hsic_lkselect import IndpTest_LKSelect_GauLap
from Test.hsic_kselect import IndpTest_KSelect
from Test.dime_gaussian import IndpTest_DIME
import argparse
import wandb

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats as stats

# import cupy as cp
import cupyx.scipy 



parser = argparse.ArgumentParser()
parser.add_argument('-e', '--experiment-name', required=True, help='Experiment name for saved files', type=str)
parser.add_argument('-dime_perm', '--dime_perm', required = False, default = 10, type = int)
parser.add_argument('-epochs', '--epochs', required = False, default = 200, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.07, type = float)
parser.add_argument('-batch_size', '--batch_size', required = False, default = 100, type = int)
parser.add_argument('-grid_search_min', '--grid_search_min', required = False, default = -2, type = int)
parser.add_argument('-grid_search_max', '--grid_search_max', required = False, default = 4, type = int)

args = parser.parse_args()

wandb.init(project='independence_testing', name=args.experiment_name)

def sinedependence(n,d,seed = 0):
    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, n)
    Z = np.random.randn(n)
    Y = 20*np.sin(4*np.pi*(X[:,0]**2 + X[:,1]**2))+Z 
    return X,Y

def run():
    repetitions = 50
    test_num = repetitions
    seed = 0 
    device = torch.device('cuda')
    d = 3
    n = 600
    test_power = np.zeros([test_num])
    
    
    for j in range(test_num):
        print('sample size:', n, 'repetition: ', j)
        X, Y = sinedependence(n, d, seed)
        Y = Y.reshape(-1,1)
        X_tensor, Y_tensor = torch.tensor(X, device=device), torch.tensor(Y,device=device)

        # alpha = 1.0, von Neumann Entropies

        dime_estimator = IndpTest_DIME( X_tensor, Y_tensor ,
                                        alpha = 1.0, isotropic = False, 
                                        dime_perm = args.dime_perm , lr = args.lr,
                                        epochs = args.epochs, batch_size = args.batch_size,
                                        grid_search_min = args.grid_search_min,
                                        grid_search_max = args.grid_search_max)
        results_dime = dime_estimator.perform_test()
        test_power[j] = float(results_dime['h0_rejected'])

        seed += 1
        # average test power
        # compute the average test power
        avg_test_power = np.mean(test_power[:j+1])
        wandb.log({"avg_test_power": avg_test_power})  


if __name__ == "__main__":
    run()
