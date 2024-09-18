
import torch
import numpy as np

# imports from representation-itl library
import repitl.kernel_utils as ku
import repitl.matrix_itl as itl
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
import torch.distributions as dist
import copy
from matplotlib import pyplot as plt
from Test.kernels import rbf_kernel, kernel_midwidth_rbf, rbf_kernel_weight

class IndpTest_DIME():
    def __init__(self, X,Y, dime_perm, 
                 alpha = 1.0, type_bandwidth = 'diagonal', 
                 epochs = 200, lr = 0.01,  split_ratio = 0.5, 
                 batch_size = None, 
                 grid_search_min = -1, grid_search_max = 3, 
                 optimizer = 'adam', 
                 scheduler = False):
        """
        X: numpy array of shape (n_samples, n_features)
        Y: numpy array of shape (n_samples, n_features)
        dime_perm: number of permutations to estimate the DiME
        alpha: Rényi entropy parameter
        type_bandwidth: 'isotropic', 'diagonal', 'anisotropic'
        epochs: number of epochs
        lr: learning rate
        split_ratio: ratio of splitting the dataset into train/test
        batch_size: batch size
        grid_search_min: minimum value of the grid search
        grid_search_max: maximum value of the grid search
        """
        assert type_bandwidth in ['isotropic', 'diagonal', 'anisotropic', 'weighted'], f"Invalid type_bandwidth: {type_bandwidth}"
        self.X = X
        self.Y = Y
        self.dime_perm = dime_perm
        self.alpha = alpha
        self.type_bandwidth = type_bandwidth
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.grid_search_min = grid_search_min
        self.grid_search_max = grid_search_max
        self.optimizer = optimizer
        self.scheduler = scheduler
    def perform_test(self, significance = 0.05, permutations = 100, seed = 0): # SEED = 0
        ### split the datasets ###
        Xtr, Ytr, Xte, Yte = self.split_samples()
        dime_null = []
        self.fit(Xtr, Ytr, epochs = self.epochs, lr = self.lr, batch_size=self.batch_size, optimizer=self.optimizer, scheduler=self.scheduler)
        # dime = self.forward_normalized(Xte,Yte, seed = seed)
        # dime = self.forward(Xte,Yte, seed = seed)
        Hxy = self.forward_joint_entropy(Xte, Yte)
        Hxy_null = []

        # Set a different seed for permutations
        perm_seed = seed 
        for i in range(permutations):
            torch.manual_seed(perm_seed)
            idx_perm = torch.randperm(Xte.shape[0])
            X_perm = Xte[idx_perm, :]
            # Y_perm = Yte[idx_perm, :]
            # dime_null.append(self.forward_normalized(Xte, Y_perm, seed = seed))
            # dime_null.append(self.forward_normalized(X_perm, Yte, seed = seed))
            # dime_null.append(self.forward(X_perm, Yte, seed = seed))
            Hxy_null.append(self.forward_joint_entropy(X_perm, Yte))
            perm_seed += 1 
        # dime_null = torch.tensor(dime_null)
        Hxy_null = torch.tensor(Hxy_null)
        dime_null =  Hxy_null.mean() - Hxy_null
        dime = Hxy_null.mean() - Hxy
        plt.hist(dime_null.cpu().numpy(), bins = 20)
        thr_dime = torch.quantile(dime_null, (1 - significance))
        
        h0_rejected = (dime>thr_dime)
        
        results_all = {}
        results_all["thresh"] = thr_dime
        results_all["test_stat"] = dime
        results_all["h0_rejected"] = h0_rejected
        
        return results_all

    def fit(self, X, Y, epochs = 200, lr = 0.01, batch_size = None, verbose = True, optimizer = 'adam', scheduler = False): 
        sigma_x, sigma_y = self.grid_search_init(X, Y)
        print('sigma_x: {}, sigma_y: {}'.format(sigma_x, sigma_y))
        if self.type_bandwidth == 'isotropic':
            sigma_x = torch.tensor(sigma_x, device = X.device)
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_x = torch.log(sigma_x).clone().detach().requires_grad_(True)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            # optimizer = torch.optim.Adam([log_sigma_x, log_sigma_y], lr=lr)
            params = [log_sigma_x, log_sigma_y]
        elif self.type_bandwidth == 'diagonal':
            sigma_x = torch.tensor(sigma_x, device = X.device)
            sigma_y = torch.tensor(sigma_y, device = X.device)
            sigma_x_diag_vals = torch.ones(X.shape[1], device=X.device, dtype=X.dtype)*sigma_x
            sigma_y_diag_vals = torch.ones(Y.shape[1], device=Y.device, dtype=Y.dtype)*sigma_y
            log_sigma_x_diag_vals = torch.log(sigma_x_diag_vals).clone().detach().requires_grad_(True)
            log_sigma_y_diag_vals = torch.log(sigma_y_diag_vals).clone().detach().requires_grad_(True)
            # optimizer = torch.optim.Adam([log_sigma_x_diag_vals, log_sigma_y_diag_vals], lr=lr)
            params = [log_sigma_x_diag_vals, log_sigma_y_diag_vals]
        elif self.type_bandwidth == 'anisotropic':
            sigma_x = sigma_x*torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
            A = sigma_x.inverse().clone().detach().requires_grad_(True) # matrix to perform metric transformation 
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            # optimizer = torch.optim.Adam([A, log_sigma_y], lr=lr)
            params = [A, log_sigma_y]
        elif self.type_bandwidth == 'weighted':
            sigma_x = torch.tensor(sigma_x, device = X.device)
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_x = torch.log(sigma_x).clone().detach().requires_grad_(True)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            attx_init = [0.0] * X.shape[1]
            atty_init = [0.0] * Y.shape[1]
            att_x = torch.tensor([attx_init],requires_grad=True, device = X.device)
            att_y = torch.tensor([atty_init],requires_grad=True, device = Y.device)
            params = [att_x, att_y, log_sigma_x, log_sigma_y]
        else:
            raise ValueError("Unsupported type_bandwidth")

        if optimizer == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr)
        elif optimizer == 'sgd':
            optimizer = torch.optim.SGD(params, lr=lr, momentum=0.2)
        else:
            raise ValueError("Unsupported optimizer type: {}".format(optimizer))
        seed = 0
        # set seed
        # Initialize the cosine annealing scheduler with a minimum learning rate
        if scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min = 0.1*lr)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        n_samples = X.shape[0]
        batch_size = n_samples if batch_size is None else batch_size
        n_batches = n_samples // batch_size

        for i in range(epochs):
            # shuffle the data
            p = torch.randperm(n_samples)
            X = X[p]
            Y = Y[p]
            optimizer.zero_grad()
            for j in range(n_batches):
                start = j * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                Y_batch = Y[start:end]
                
                if self.type_bandwidth == 'isotropic':
                    sigma_x = torch.exp(log_sigma_x)
                    sigma_y = torch.exp(log_sigma_y)
                    Kx = ku.gaussianKernel(X_batch, X_batch, sigma_x)
                    Ky = ku.gaussianKernel(Y_batch, Y_batch, sigma_y)
                elif self.type_bandwidth == 'diagonal':
                    sigma_x_diag_vals = torch.exp(log_sigma_x_diag_vals)
                    sigma_y_diag_vals = torch.exp(log_sigma_y_diag_vals)
                    X_weighted = X_batch/sigma_x_diag_vals  # performs column-wise division, equivalent to X_batch @ diag(1/sigma_x_diag_vals)
                    Y_weighted = Y_batch/sigma_y_diag_vals
                    Kx = ku.gaussianKernel(X_weighted, X_weighted, 1)
                    Ky = ku.gaussianKernel(Y_weighted, Y_weighted, 1)
                elif self.type_bandwidth == 'anisotropic':
                    sigma_y = torch.exp(log_sigma_y)
                    X_ = X_batch @ A
                    Kx = ku.gaussianKernel(X_, X_, 1)
                    Ky = ku.gaussianKernel(Y_batch, Y_batch, sigma_y)
                else:
                    sigma_x = torch.exp(log_sigma_x)
                    sigma_y = torch.exp(log_sigma_y)
                    weight_x = torch.sigmoid(att_x)
                    weight_y = torch.sigmoid(att_y)
                    Kx, Ky = self.cal_weight_kernels(X_batch, Y_batch, sigma_x, sigma_y, weight_x, weight_y)
                
                # mi = -1 * dime_normalized(Kx, Ky, alpha=self.alpha, n_iters=self.dime_perm, seed=seed)
                mi = -1 * dime(Kx, Ky, alpha=self.alpha, n_iters=self.dime_perm, seed=seed)
                mi.backward()
                optimizer.step()
                seed += 1
            if scheduler:
                scheduler.step()
            if verbose and i % 10 == 0:
                print('Iteration: {}, DiME: {}'.format(i, -1 * mi.item()))
                # print(sigma_x_diag_vals)
                print(weight_x, weight_y, sigma_x, sigma_y)
        if self.type_bandwidth == 'isotropic':
            sigma_x = torch.exp(log_sigma_x)
            sigma_y = torch.exp(log_sigma_y)
            self.sigma_x = sigma_x.detach().item()
            self.sigma_y = sigma_y.detach().item()
        elif self.type_bandwidth == 'diagonal':
            sigma_x_diag_vals = torch.exp(log_sigma_x_diag_vals)
            sigma_y_diag_vals = torch.exp(log_sigma_y_diag_vals)
            self.sigma_x_diag_vals = sigma_x_diag_vals.detach()
            self.sigma_y_diag_vals = sigma_y_diag_vals.detach()
        elif self.type_bandwidth == 'anisotropic':
            self.A = A.detach()
            sigma_y = torch.exp(log_sigma_y)
            self.sigma_y = sigma_y.detach().item()
        else:
            sigma_x = torch.exp(log_sigma_x)
            sigma_y = torch.exp(log_sigma_y)
            self.sigma_x = sigma_x.detach().item()
            self.sigma_y = sigma_y.detach().item()
            self.att_x = att_x.detach()
            self.att_y = att_y.detach()
    def grid_search_init(self, X, Y):
        """
        Using grid_search to init the widths (similar to HSIC)
        """
        n_bandwiths = 5
        pairwise_matrix_x = torch.cdist(X, X, p = 2)  # l1 and l2 distances
        pairwise_matrix_y = torch.cdist(Y, Y, p = 2)  # l1 and l2 distances
        # Collection of bandwidths
        def compute_bandwidths(distances, n_bandwiths):
            median = torch.median(distances[distances>0])
            bandwidths = torch.sqrt(0.5*median)*2**torch.linspace(self.grid_search_min, self.grid_search_max, n_bandwiths, device=X.device) # torch.linspace(-3, 3, n_bandwiths, device=X.device)
            return bandwidths

        triu_indices = torch.triu_indices(pairwise_matrix_x.shape[0],pairwise_matrix_x.shape[0], offset=0)
        distances_x = pairwise_matrix_x[triu_indices[0], triu_indices[1]]
        distances_y = pairwise_matrix_y[triu_indices[0], triu_indices[1]]
        bandwidths_x = compute_bandwidths(distances_x, n_bandwiths)
        bandwidths_y = compute_bandwidths(distances_y, n_bandwiths) 
        dime_pair = []
        sigma_pair = []
        for i, sigma_x in enumerate(bandwidths_x):
            for j, sigma_y in enumerate(bandwidths_y):

                Kx = ku.gaussianKernel(X,X, sigma_x)
                Ky = ku.gaussianKernel(Y,Y, sigma_y)
                mi = dime_normalized(Kx,Ky,alpha=self.alpha, n_iters=self.dime_perm, seed = 0) # 2 times to have a good estimate of DiME 
                dime_pair.append(mi.item())
                sigma_pair.append((sigma_x.item(), sigma_y.item()))

        dime_array = np.array(dime_pair)
        indm = np.argmax(dime_array)
        print('Max DiME_n: {}'.format(dime_array[indm]))
        return sigma_pair[indm]

    def split_samples(self):
        """
        split datasets into train/test datasets
        """
        n = len(self.X)
        p = np.random.permutation(n)
        tr_size = int(n*self.split_ratio)
        ind_train = p[:tr_size]
        ind_test = p[tr_size:]
        
        Xtr = self.X[ind_train,:]
        Ytr = self.Y[ind_train,:]
        Xte = self.X[ind_test,:]
        Yte = self.Y[ind_test,:]
        
        if len(Xtr.size())==1:
            Xtr = Xtr.reshape(-1,1)
        if len(Ytr.size())==1:
            Ytr = Ytr.reshape(-1,1)
        if len(Xte.size())==1:
            Xte = Xte.reshape(-1,1)
        if len(Yte.size())==1:
            Yte = Yte.reshape(-1,1)
        
        return Xtr, Ytr, Xte, Yte

    def forward(self, X, Y, seed = None):
        with torch.no_grad():
            if self.type_bandwidth == 'isotropic':
                Kx = ku.gaussianKernel(X,X, self.sigma_x)
                Ky = ku.gaussianKernel(Y,Y, self.sigma_y)
            elif self.type_bandwidth == 'diagonal':
                X_weighted = X/self.sigma_x_diag_vals
                Y_weighted = Y/self.sigma_y_diag_vals
                Kx = ku.gaussianKernel(X_weighted, X_weighted, 1)
                Ky = ku.gaussianKernel(Y_weighted, Y_weighted, 1)
            elif self.type_bandwidth == 'anisotropic':
                X_ = X @ self.A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,self.sigma_y)
            else:
                weight_x = torch.sigmoid(self.att_x)
                weight_y = torch.sigmoid(self.att_y)
                Kx, Ky = self.cal_weight_kernels(X, Y, self.sigma_x, self.sigma_y, weight_x, weight_y)
            mi = dime(Kx,Ky, alpha=self.alpha, n_iters=self.dime_perm, seed = seed)
            
        return mi
    def forward_normalized(self, X, Y, seed = None, cap = False):
        with torch.no_grad():
            if self.type_bandwidth == 'isotropic':
                Kx = ku.gaussianKernel(X,X, self.sigma_x)
                Ky = ku.gaussianKernel(Y,Y, self.sigma_y)
            elif self.type_bandwidth == 'diagonal':
                X_weighted = X/self.sigma_x_diag_vals
                Y_weighted = Y/self.sigma_y_diag_vals
                Kx = ku.gaussianKernel(X_weighted, X_weighted, 1)
                Ky = ku.gaussianKernel(Y_weighted, Y_weighted, 1)
            elif self.type_bandwidth == 'anisotropic':
                X_ = X @ self.A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,self.sigma_y)
            else:
                weight_x = torch.sigmoid(self.att_x)
                weight_y = torch.sigmoid(self.att_y)
                Kx, Ky = self.cal_weight_kernels(X, Y, self.sigma_x, self.sigma_y, weight_x, weight_y)
            mi = dime_normalized(Kx,Ky, alpha=self.alpha, n_iters=self.dime_perm, seed = seed)
        return mi
    def forward_joint_entropy(self, X, Y):
        with torch.no_grad():
            if self.type_bandwidth == 'isotropic':
                Kx = ku.gaussianKernel(X,X, self.sigma_x)
                Ky = ku.gaussianKernel(Y,Y, self.sigma_y)
            elif self.type_bandwidth == 'diagonal':
                X_weighted = X/self.sigma_x_diag_vals
                Y_weighted = Y/self.sigma_y_diag_vals
                Kx = ku.gaussianKernel(X_weighted, X_weighted, 1)
                Ky = ku.gaussianKernel(Y_weighted, Y_weighted, 1)
            elif self.type_bandwidth == 'anisotropic':
                X_ = X @ self.A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,self.sigma_y)
            else:
                weight_x = torch.sigmoid(self.att_x)
                weight_y = torch.sigmoid(self.att_y)
                Kx, Ky = self.cal_weight_kernels(X, Y, self.sigma_x, self.sigma_y, weight_x, weight_y)
            Hxy = joint_entropy(Kx,Ky, alpha=self.alpha)
        return Hxy
    def forward_mi(self, X, Y, seed = None):
        with torch.no_grad():
            if self.type_bandwidth == 'isotropic':
                Kx = ku.gaussianKernel(X,X, self.sigma_x)
                Ky = ku.gaussianKernel(Y,Y, self.sigma_y)
            elif self.type_bandwidth == 'diagonal':
                X_weighted = X/self.sigma_x_diag_vals
                Y_weighted = Y/self.sigma_y_diag_vals
                Kx = ku.gaussianKernel(X_weighted, X_weighted, 1)
                Ky = ku.gaussianKernel(Y_weighted, Y_weighted, 1)
            elif self.type_bandwidth == 'anisotropic':
                X_ = X @ self.A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,self.sigma_y)
            else:
                weight_x = torch.sigmoid(self.att_x)
                weight_y = torch.sigmoid(self.att_y)
                Kx, Ky = self.cal_weight_kernels(X, Y, self.sigma_x, self.sigma_y, weight_x, weight_y)
            mi = mutual_information(Kx, Ky, alpha=self.alpha)
        return mi

    def cal_weight_kernels(self, X, Y, kernel_width_x, kernel_width_y, weight_x, weight_y):
        """
        Calculate (weighted rbf) kernels
        """
        K = rbf_kernel_weight(X, X, kernel_width_x, weight_x)
        L = rbf_kernel_weight(Y, Y, kernel_width_y, weight_y)
        
        return K, L
    
    def midwidth_rbf(self, X, Y):
        """
        Calculate midwidth of Gaussian kernels 
        (also return maxwidth that can be used to limit the range in learning kernels)
        
        Return 
        wx_mid, wy_mid, wx_max, wy_max
        """
        wx_mid, wy_mid, wx_max, wy_max = kernel_midwidth_rbf(X, Y)
        
        return wx_mid, wy_mid, wx_max, wy_max    

import torch
import repitl.matrix_itl as itl

def permuteGram(K):
    """
    Randomly permutes the rows and columns of a square matrix
    """
    
    assert K.shape[0] == K.shape[1], f"matrix dimensions must be the same"
    idx = torch.randperm(K.shape[0])
    K = K[idx, :]
    K = K[:, idx]
    return K

def dime(Kx, Ky, alpha, n_iters=1, shouldReturnComponents = False, seed = None):
    """
    Computes the difference of entropy equation of the following form. Let P be a random permutation matrix
    
    doe(Kx, Ky) = EXPECTATION[ H_alpha(Kx, P Ky P) - H_alpha(Kx, Ky)]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    if alpha != 1:
        H = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha)
    else:
        H = vonNeumannEntropy(Kx*Ky)
    
    H_perm_avg = 0
    for i in range(n_iters):
        if alpha != 1:
            H_perm = itl.matrixAlphaJointEntropy([Kx, permuteGram(Ky)], alpha=alpha)
        else:
            H_perm = vonNeumannEntropy(Kx*permuteGram(Ky))

        H_perm_avg = H_perm_avg + (H_perm / n_iters)
    
    if shouldReturnComponents:
        return H_perm_avg - H, H, H_perm_avg
    
    return H_perm_avg - H

def dime_normalized(Kx, Ky, alpha, n_iters=1, seed = None, cap = False):
    """
    Computes the difference of entropy equation of the following form. Let P be a random permutation matrix
    
    doe(Kx, Ky) = EXPECTATION[ H_alpha(Kx, P Ky P) - H_alpha(Kx, Ky)]/STD[ H_alpha(Kx, P Ky P)]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    if alpha != 1:
        H = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha)
    else:
        H = vonNeumannEntropy(Kx*Ky)
    
    H_perm_ = []
    for i in range(n_iters):
        if alpha != 1:
            H_perm = itl.matrixAlphaJointEntropy([Kx, permuteGram(Ky)], alpha=alpha)
        else:
            H_perm = vonNeumannEntropy(Kx*permuteGram(Ky))
        H_perm_.append(H_perm)

        # compute mean and standard deviation of H_perm_
    H_perm = torch.stack(H_perm_)
    H_perm_avg = torch.mean(H_perm)
    H_perm_std = torch.std(H_perm)
    dime_normalized = (H_perm_avg - H) / (H_perm_std + 1e-15)
    if cap and dime_normalized > 100: # re run with a different seed if the value is too high (this implies low variance, usually a bad permutation)
        seed += 1 if seed is not None else None
        dime_normalized = dime(Kx, Ky, alpha, n_iters=n_iters, seed = seed)
    return dime_normalized


def joint_entropy(Kx, Ky, alpha):
    """
    Computes the joint entropy of two kernels
    """
    if alpha != 1:
        H = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha)
    else:
        H = vonNeumannEntropy(Kx*Ky)
    
    return H

def dime_max(Kx, Ky, alpha, n_iters=1, shouldReturnComponents = False, seed = None):
    """
    Computes the difference of entropy equation of the following form. Let P be a random permutation matrix
    
    doe(Kx, Ky) = EXPECTATION[ H_alpha(Kx, P Ky P) - H_alpha(Kx, Ky)]
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    if alpha != 1:
        H = itl.matrixAlphaJointEntropy([Kx, Ky], alpha=alpha)
    else:
        H = vonNeumannEntropy(Kx*Ky)
    
    H_perm_ = []
    for i in range(n_iters):
        if alpha != 1:
            H_perm = itl.matrixAlphaJointEntropy([Kx, permuteGram(Ky)], alpha=alpha)
        else:
            H_perm = vonNeumannEntropy(Kx*permuteGram(Ky))

        H_perm_.append(H_perm)
    
    H_perm_max = torch.max(torch.stack(H_perm_))
    
    if shouldReturnComponents:
        return H_perm_max - H, H, H_perm_max
    
    return H_perm_max - H

def mutual_information(Kx, Ky, alpha):
    """
    Computes the kernel based mutual information
    """
    if alpha != 1:
        MI = itl.matrixAlphaMutualInformation(Kx, Ky, alpha)
    else:
        Hx =vonNeumannEntropy(Kx)
        Hy = vonNeumannEntropy(Ky)
        Hxy = vonNeumannEntropy(Kx*Ky)
        MI = Hx + Hy - Hxy
    return MI


def vonNeumannEntropy(K, normalize = True, rank = None, retrieve_rank = False):
    ek, _ = torch.linalg.eigh(K)
    if rank is None:
        N = len(ek)
        lambda1 = ek[-1] # Largest eigenvalue
        rtol = lambda1*N*torch.finfo(ek.dtype).eps
        mk = torch.gt(ek, 0)
        mek = ek[mk]
    elif rank < K.shape[0]:
        ek_lr = torch.zeros_like(ek)
        ek_lr[-rank:] = ek[-rank:]
        ek_lr = ek_lr/ek_lr.sum() 
        mk = torch.gt(ek_lr, 0)
        mek = ek_lr[mk]

    if normalize:
        mek = mek/mek.sum()   
    H = -1*torch.sum(mek*torch.log(mek))
    if retrieve_rank:
        rank = compute_rank(ek)
        return H, rank
    return H
def compute_rank(eigenvalues):
    # Similar to pytorch implementation
    N = len(eigenvalues)
    eigenvalues = torch.abs(eigenvalues)
    lambda1 = eigenvalues[-1] # Largest eigenvalue
    rtol = N*torch.finfo(eigenvalues.dtype).eps
    rank = torch.sum(eigenvalues > rtol*lambda1)
    return rank
# Taylor expansion
def matrix_log(Q, order=4):
    n = Q.shape[0]
    Q = Q - torch.eye(n).detach().to(Q.device)
    cur = Q
    res = torch.zeros_like(Q).detach().to(Q.device)
    for k in range(1, order + 1):
        if k % 2 == 1:
            res = res + cur * (1. / float(k))
        else:
            res = res - cur * (1. / float(k))
        cur = cur @ Q
    return res

def vonNeumannEntropy_approx(K, order=4):
    K = K/K.trace()
    return torch.trace(- K @ matrix_log(K, order))

# def permuteGram(K, seed = None):
#     """
#     Randomly permutes the rows and columns of a square matrix
#     """
#     if seed is not None:
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)

#     assert K.shape[0] == K.shape[1], f"matrix dimensions must be the same"
#     idx = torch.randperm(K.shape[0])
#     K = K[idx, :]
#     K = K[:, idx]
#     return K













