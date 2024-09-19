from matplotlib import pyplot as plt
import numpy as np
import h5py
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
parser.add_argument('-epochs', '--epochs', required = False, default = 300, type = int)
parser.add_argument('-lr', '--lr', required = False, default = 0.05, type = float)
parser.add_argument('-batch_size', '--batch_size', required = False, default = None, type = int)
parser.add_argument('-grid_search_min', '--grid_search_min', required = False, default = -3, type = int)
parser.add_argument('-grid_search_max', '--grid_search_max', required = False, default = 3, type = int)

args = parser.parse_args()



_FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                     'orientation']
_NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                          'scale': 8, 'shape': 4, 'orientation': 15}
# methods for sampling unconditionally/conditionally on a given factor
def get_index(factors):
  """ Converts factors to indices in range(num_data)
  Args:
    factors: np array shape [6,batch_size].
             factors[i]=factors[i,:] takes integer values in 
             range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

  Returns:
    indices: np array shape [batch_size].
  """
  indices = 0
  base = 1
  for factor, name in reversed(list(enumerate(_FACTORS_IN_ORDER))):
    indices += factors[factor] * base
    base *= _NUM_VALUES_PER_FACTOR[name]
  return indices


def sample_batch_spheres(images, batch_size, fixed_factor_str = 'shape', fixed_factor_value = 2, seed = 0):
  """ Samples a batch of images of spheres, with
      the other factors varying randomly.
  Args:
    batch_size: number of images to sample.
    fixed_factor: shape.
    fixed_factor_value: 2 for spheres. 
      in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).

  Returns:
    X: images shape [batch_size,64*64*3]
    Y: orientation [batch_size]

  """
  np.random.seed(seed)
  fixed_factor = _FACTORS_IN_ORDER.index(fixed_factor_str)  # floor hue
  factors = np.zeros([len(_FACTORS_IN_ORDER), batch_size],
                     dtype=np.int32)
  for factor, name in enumerate(_FACTORS_IN_ORDER):
    num_choices = _NUM_VALUES_PER_FACTOR[name]
    factors[factor] = np.random.choice(num_choices, batch_size)
  factors[fixed_factor] = fixed_factor_value
  Y = factors[5]
  indices = get_index(factors)
  ims = []
  for ind in indices:
    im = images[ind]
    im = np.asarray(im)
    ims.append(im)
  ims = np.stack(ims, axis=0)
  ims = ims / 255. # normalise values to range [0,1]
  ims = ims.astype(np.float32)
  ims.reshape([batch_size, 64, 64, 3])
  X = ims.reshape([batch_size, 64*64*3])
  return X,Y


def run():
    # load dataset
    dataset = h5py.File('3dshapes.h5', 'r')
    print(dataset.keys())
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

    repetitions = 100
    fname = args.DATAFOLDER + '/' + str(args.experiment_name) + str(args.repId) + '.npz'
    if args.parallel:
        test_num = 1
        seed = 0 + args.repId - 1 
    else:
        test_num = repetitions
        seed = 0 
    # device = torch.device('cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sample_sizes = (64,)
    n_samples = len(sample_sizes)
    n_tests = 9
    test_power = np.zeros([n_tests,n_samples, test_num])
    

    for i, n in enumerate(sample_sizes):
        for j in range(test_num):
            print('sample size:', n, 'repetition: ', j)
            X, Y = sample_batch_spheres(images, batch_size = n, seed=seed)
            Y = Y.reshape(-1,1)
            Y = Y.astype(np.float32)
            X_tensor, Y_tensor = torch.tensor(X, device=device), torch.tensor(Y,device=device)

            # alpha = 1.0, von Neumann Entropies
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 1.0, type_bandwidth= 'isotropic',
                                            dime_perm = args.dime_perm , lr = args.lr,
                                            epochs = args.epochs, batch_size = args.batch_size,
                                            grid_search_min = args.grid_search_min,
                                            grid_search_max = args.grid_search_max)
            results_dime = dime_estimator.perform_test()
            test_power[0, i, j] = float(results_dime['h0_rejected'])
            
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 1.0,  type_bandwidth= 'weighted',
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
            
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 2.0, type_bandwidth= "weighted", 
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
        
            dime_estimator = IndpTest_DIME( X_tensor, Y_tensor, alpha = 0.5, type_bandwidth= "weighted",
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