
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

class IndpTest_DIME():
    def __init__(self, X,Y, dime_perm, alpha = 1.0, isotropic = True, epochs = 100, lr = 0.01, approx = False, order_approx = 4,  split_ratio = 0.5):
        self.X = X
        self.Y = Y
        self.dime_perm = dime_perm
        self.alpha = alpha
        self.approx = approx
        self.order_approx = order_approx
        self.isotropic = isotropic
        self.sigma_inverse = None
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.lr = lr

    def fit(self, X, Y, epochs = 100, lr = 0.001, verbose = False):
        if self.isotropic:
            sigma_x, sigma_y = self.grid_search_init(X, Y)
            sigma_x = torch.tensor(sigma_x, device = X.device)
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_x = torch.log(sigma_x).clone().detach().requires_grad_(True)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([log_sigma_x, log_sigma_y], lr=lr)
        else:
            sigma_x, sigma_y = self.grid_search_init(X, Y)
            sigma_x = sigma_x*torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
            A = sigma_x.inverse().clone().detach().requires_grad_(True) # matrix to perform metric transformation 
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([A, log_sigma_y], lr=lr)
        seed = 0
        for i in range(epochs):
            optimizer.zero_grad()
            if self.isotropic:
                sigma_x = torch.exp(log_sigma_x)
                sigma_y = torch.exp(log_sigma_y)
                Kx = ku.gaussianKernel(X,X, sigma_x)
                Ky = ku.gaussianKernel(Y,Y, sigma_y)
            else:
                sigma_y = torch.exp(log_sigma_y)
                X_ = X @ A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,sigma_y)
            # change seed every l epochs
            seed = seed +  i // 100
            mi = -1*dime(Kx,Ky,alpha=self.alpha, n_iters=self.dime_perm, seed = seed)
            # if mi is positive don't update the parameters
            if mi > 0:
                mi = torch.tensor(0.0, device = X.device)
                seed += 1
            else: 
                mi.backward()
                optimizer.step()

            if verbose and i % 99 == 0:
                # print('sigma_x: {}, sigma_y: {}'.format(sigma_x.item(), sigma_y.item()))
                print('Iteration: {}, DiME: {}'.format(i,-1*mi.item()))
        if self.isotropic:
            self.sigma_x = sigma_x.detach().item()
            self.sigma_y = sigma_y.detach().item()
        else:
            self.A = A.detach()
            self.sigma_y = sigma_y.detach().item()
    def grid_search_init(self, X, Y):
        """
        Using grid_search to init the widths (similar to HSIC)
        """
        n_bandwiths = 5
        pairwise_matrix_x = torch.cdist(X, X, p = 2)  # l1 and l2 distances
        pairwise_matrix_y = torch.cdist(Y, Y, p = 2)  # l1 and l2 distances
        # Collection of bandwidths
        def compute_bandwidths(distances, n_bandwiths):
            median = torch.median(distances)
            bandwidths = median*2**torch.linspace(-3, 3, n_bandwiths, device=X.device)
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
                mi = dime(Kx,Ky,alpha=self.alpha, n_iters=self.dime_perm, seed = 0)
                dime_pair.append(mi.item())
                sigma_pair.append((sigma_x.item(), sigma_y.item()))

        dime_array = np.array(dime_pair)
        indm = np.argmax(dime_array)

        return sigma_pair[indm]

    def perform_test(self, significance = 0.05, permutations = 100, seed = 0):
        torch.manual_seed(seed)
                ### split the datasets ###
        Xtr, Ytr, Xte, Yte = self.split_samples()
        dime_null = []
        self.fit(Xtr, Ytr, epochs = self.epochs, lr = self.lr)
        for i in range(permutations):
            X_perm = Xte[torch.randperm(Xte.shape[0])]
            dime_null.append(self.forward(X_perm, Yte, seed = seed)) 
        dime_null = torch.tensor(dime_null)
        thr_dime = torch.quantile(dime_null, (1 - significance))
        dime = self.forward(Xte,Yte, seed=seed)
        h0_rejected = (dime>thr_dime)
        
        results_all = {}
        results_all["thresh"] = thr_dime
        results_all["test_stat"] = dime
        results_all["h0_rejected"] = h0_rejected
        
        return results_all

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
            if self.isotropic:
                Kx = ku.gaussianKernel(X,X, self.sigma_x)
                Ky = ku.gaussianKernel(Y,Y, self.sigma_y)
            else:
                X_ = X @ self.A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,self.sigma_y)
            mi = dime(Kx,Ky, alpha=self.alpha, n_iters=self.dime_perm, seed = seed)
        return mi

    

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

def permuteGram(K, seed = None):
    """
    Randomly permutes the rows and columns of a square matrix
    """
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    assert K.shape[0] == K.shape[1], f"matrix dimensions must be the same"
    idx = torch.randperm(K.shape[0])
    K = K[idx, :]
    K = K[:, idx]
    return K













