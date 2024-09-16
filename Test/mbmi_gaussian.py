
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

class IndpTest_kernelMI():
    def __init__(self, X,Y, alpha = 1.0, isotropic = True, epochs = 200, lr = 0.01,  split_ratio = 0.5, batch_size = None):
        self.X = X
        self.Y = Y
        self.alpha = alpha
        self.isotropic = isotropic
        self.split_ratio = split_ratio
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
    def perform_test(self, significance = 0.05, permutations = 100, seed = 0): # SEED = 0
        ### split the datasets ###
        Xtr, Ytr, Xte, Yte = self.split_samples()
        mutual_information_null = []
        self.fit(Xtr, Ytr, epochs = self.epochs, lr = self.lr, batch_size=self.batch_size)
        mutual_information = self.forward(Xte,Yte, seed = 0)

        # Set a different seed for permutations
        perm_seed = seed 
        for i in range(permutations):
            torch.manual_seed(perm_seed)
            idx_perm = torch.randperm(Xte.shape[0])
            X_perm = Xte[idx_perm, :]
            mutual_information_null.append(self.forward(X_perm, Yte, seed = seed))
            perm_seed += 1 
        mutual_information_null = torch.tensor(mutual_information_null)
        plt.hist(mutual_information_null.cpu().numpy(), bins = 20)
        thr_mutual_information = torch.quantile(mutual_information_null, (1 - significance))
        
        h0_rejected = (mutual_information>thr_mutual_information)
        
        results_all = {}
        results_all["thresh"] = thr_mutual_information
        results_all["test_stat"] = mutual_information
        results_all["h0_rejected"] = h0_rejected
        
        return results_all

    def fit(self, X, Y, epochs = 200, lr = 0.01, batch_size = None, verbose = True): # changed learning rate
        sigma_x, sigma_y = self.grid_search_init(X, Y)
        print('sigma_x: {}, sigma_y: {}'.format(sigma_x, sigma_y))
        if self.isotropic:
            sigma_x = torch.tensor(sigma_x, device = X.device)
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_x = torch.log(sigma_x).clone().detach().requires_grad_(True)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([log_sigma_x, log_sigma_y], lr=lr)
        else:
            sigma_x = sigma_x*torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
            A = sigma_x.inverse().clone().detach().requires_grad_(True) # matrix to perform metric transformation 
            sigma_y = torch.tensor(sigma_y, device = X.device)
            log_sigma_y = torch.log(sigma_y).clone().detach().requires_grad_(True)
            optimizer = torch.optim.Adam([A, log_sigma_y], lr=lr)
        n_samples = X.shape[0]
        batch_size = n_samples if batch_size is None else batch_size
        n_batches = n_samples // batch_size

        for i in range(epochs):
            optimizer.zero_grad()
            for j in range(n_batches):
                start = j * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                Y_batch = Y[start:end]
                
                if self.isotropic:
                    sigma_x = torch.exp(log_sigma_x)
                    sigma_y = torch.exp(log_sigma_y)
                    Kx = ku.gaussianKernel(X_batch, X_batch, sigma_x)
                    Ky = ku.gaussianKernel(Y_batch, Y_batch, sigma_y)
                else:
                    sigma_y = torch.exp(log_sigma_y)
                    X_ = X_batch @ A
                    Kx = ku.gaussianKernel(X_, X_, 1)
                    Ky = ku.gaussianKernel(Y_batch, Y_batch, sigma_y)
                
                
                mi =  -1*kernel_mutual_information(Kx, Ky, alpha=self.alpha)
                mi.backward()
                optimizer.step()
            if verbose and i % 10 == 0:
                print('Iteration: {}, mutual information: {}'.format(i, -1 * mi.item()))
        if self.isotropic:
            sigma_x = torch.exp(log_sigma_x)
            sigma_y = torch.exp(log_sigma_y)
            self.sigma_x = sigma_x.detach().item()
            self.sigma_y = sigma_y.detach().item()
        else:
            self.A = A.detach()
            sigma_y = torch.exp(log_sigma_y)
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
            median = torch.median(distances[distances>0])
            bandwidths = torch.sqrt(0.5*median)*2**torch.linspace(-1, 3, n_bandwiths, device=X.device) # torch.linspace(-3, 3, n_bandwiths, device=X.device)
            return bandwidths

        triu_indices = torch.triu_indices(pairwise_matrix_x.shape[0],pairwise_matrix_x.shape[0], offset=0)
        distances_x = pairwise_matrix_x[triu_indices[0], triu_indices[1]]
        distances_y = pairwise_matrix_y[triu_indices[0], triu_indices[1]]
        bandwidths_x = compute_bandwidths(distances_x, n_bandwiths)
        bandwidths_y = compute_bandwidths(distances_y, n_bandwiths) 
        mutual_information_pair = []
        sigma_pair = []
        for i, sigma_x in enumerate(bandwidths_x):
            for j, sigma_y in enumerate(bandwidths_y):

                Kx = ku.gaussianKernel(X,X, sigma_x)
                Ky = ku.gaussianKernel(Y,Y, sigma_y)
                mi = kernel_mutual_information(Kx,Ky,alpha=self.alpha) 
                mutual_information_pair.append(mi.item())
                sigma_pair.append((sigma_x.item(), sigma_y.item()))

        mutual_information_array = np.array(mutual_information_pair)
        indm = np.argmax(mutual_information_array)
        print('Max mutual_information_n: {}'.format(mutual_information_array[indm]))
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
            if self.isotropic:
                Kx = ku.gaussianKernel(X,X, self.sigma_x)
                Ky = ku.gaussianKernel(Y,Y, self.sigma_y)
            else:
                X_ = X @ self.A # metric transformation (Changing the scale in different dimensions)
                Kx = ku.gaussianKernel(X_,X_,1) # sigma is one because we have already performed the metric transformation
                Ky = ku.gaussianKernel(Y,Y,self.sigma_y)
            mi = kernel_mutual_information(Kx, Ky, alpha=self.alpha)
        return mi

    

import torch
import repitl.matrix_itl as itl



def kernel_mutual_information(Kx,Ky, alpha = 1.0):
    n = Kx.shape[0]
    ekx, _ = torch.linalg.eigh(Kx)
    mkx = torch.gt(ekx, 0.0)
    mekx = ekx[mkx]
    eky, _ = torch.linalg.eigh(Ky)
    mky = torch.gt(eky, 0.0)
    meky = eky[mky]
    # compute the top n eigenvalues of the kronecker product kernel
    mekxy = top_k_prod(mekx, meky, n)
    mekxy = torch.tensor(mekxy)
    # Compute joint kernel
    Kxy = Kx*Ky
    # compute the entropy of the top n eigenvalues of the kronecker product kernel
    if alpha == 1.0:
        HxKrony = vonNeumannEigenValues(mekxy)
        Hxy = vonNeumannEntropy(Kxy)
    else:
        HxKrony = renyiEigenValues(mekxy)
        Hxy = itl.matrixAlphaEntropy(Kxy, alpha)


    MI = HxKrony - Hxy
    return MI

import heapq
def top_k_prod(A, B, k):
    # eigenvalues are usually sorted in ascending order
    # fix this to work with ascending order and avoid flipping
    A = torch.flip(A,[0])
    B = torch.flip(B,[0])
    result = []
    heap = [(-A[i] * B[0], i, 0) for i in range(len(A))]
    while heap and len(result) < k:
        p, a, b = heapq.heappop(heap)
        result.append(-p)
        if b < len(B)-1:
            heapq.heappush(heap, (-A[a] * B[b+1], a, b+1))
            
    return result
def vonNeumannEigenValues(Ev):
    mk = torch.gt(Ev, 0.0)
    mek = Ev[mk]
    mek = mek / torch.sum(mek)
    H = -1*torch.sum(mek*torch.log(mek))
    return H

def renyiEigenValues(Ev, alpha):
    mk = torch.gt(Ev, 0.0)
    mek = Ev[mk]
    mek = mek / torch.sum(mek)
    H = 1/(1-alpha)*torch.log(torch.sum(mek**alpha))
    return H

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













