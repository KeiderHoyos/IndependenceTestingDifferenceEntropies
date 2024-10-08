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
parser.add_argument('-dataset', '--dataset', required = False, default = 'isa', type = str)
parser.add_argument('-dime_perm', '--dime_perm', required = False, default = 10, type = int)
parser.add_argument('-epochs', '--epochs', required = False, default = 200, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.07, type = float)
parser.add_argument('-batch_size', '--batch_size', required = False, default = None, type = int)
parser.add_argument('-grid_search_min', '--grid_search_min', required = False, default = -2, type = int)
parser.add_argument('-grid_search_max', '--grid_search_max', required = False, default = 4, type = int)
parser.add_argument('-scheduler', '--scheduler', required = False, default = 'False', type = str)

args = parser.parse_args()

wandb.init(project='independence_testing', name=args.experiment_name)

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

def sinedependence(n,d,seed = 0):
    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, n)
    Z = np.random.randn(n)
    Y = 20*np.sin(4*np.pi*(X[:,0]**2 + X[:,1]**2))+Z 
    return X,Y

def GSign(n,d, seed):
    np.random.seed(seed)
    mean = np.zeros(d)
    cov = np.eye(d)
    X = np.random.multivariate_normal(mean, cov, n)
    sign_X = np.sign(X)
    Z = np.random.randn(n)
    Y = np.abs(Z)*np.prod(sign_X,1)
    return X,Y

def run():
    repetitions = 50
    test_num = repetitions
    seed = 0 
    device = torch.device('cuda')
    test_power = np.zeros([test_num])

    if args.scheduler == 'True':
        args.scheduler = True
    else:   
        args.scheduler = False

    
    for j in range(test_num):
        if args.dataset == 'isa':
            n = 128
            d = 4
            sigma_normal = 0.1
            alpha = 0.6666 
            X, Y =  generate_ISA(n,d,sigma_normal,alpha, seed = seed)
        elif args.dataset == 'sinusoid':
            w = 3
            n = 500
            X, Y = Sinusoid_Generator(n, w, seed)
            Y = Y.reshape(-1,1)
        elif args.dataset == 'sine':
            d = 3
            n = 900
            X, Y = sinedependence(n, d, seed)
            Y = Y.reshape(-1,1)
        elif args.dataset == 'gsign':
            d = 4
            n = 600
            X, Y =  GSign(n,d, seed)
            Y = Y.reshape(-1,1)
        else:
            raise ValueError("Invalid dataset")

        # Y = Y.reshape(-1,1)
        X_tensor, Y_tensor = torch.tensor(X, device=device), torch.tensor(Y,device=device)

        # alpha = 1.0, von Neumann Entropies
        dime_estimator = IndpTest_DIME(X_tensor, Y_tensor, 
                                        alpha = 1.0, type_bandwidth = "weighted", 
                                        dime_perm= args.dime_perm, lr = args.lr, 
                                        grid_search_min = args.grid_search_min,
                                        grid_search_max = args.grid_search_max, 
                                        epochs=args.epochs, scheduler=args.scheduler)
        
        results_dime = dime_estimator.perform_test()
        test_power[j] = float(results_dime['h0_rejected'])

        seed += 1
        # average test power
        # compute the average test power
        avg_test_power = np.mean(test_power[:j+1])
        wandb.log({"avg_test_power": avg_test_power})  


if __name__ == "__main__":
    run()
