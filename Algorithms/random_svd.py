import numpy as np
from numpy.linalg import qr 
from numpy.typing import NDArray


def random_subspace_iter(A : NDArray, Y : NDArray, r : int) -> NDArray:
    ''' Given an mxn matrix A and an mxj matrix Y (product of AW for a Gaussian matrix W), and an integer q, 
        this algorithm computes an mxj orthonormal matrix Q whose range approximates the range of A.

            Parameters:
                A (ndrray) : mxn matrix
                Y (ndarray) : mxj matrix, product AW for a Gaussian matrix W
                r (int) : specifies desired number of iterations to perform

            Returns:
                Q (ndarray) : mxj matrix with range(Q) approximately equal to range(A)
    '''

    # Compute the QR factorization Y0 = Q0*R0
    Y_old = Y.copy()
    Q_old, R_old = qr(Y_old)

    # iterate from 1 to r
    for _ in range(1, r+1):
        Y_approx = A.T @ Q_old
        Q_approx, R_approx = qr(Y_approx)
        Y_new = A @ Q_approx 
        Q_new, R_new = qr(Y_new)

        # update values
        Q_old = Q_new.copy()
    
    # set Q = Qq
    return Q_new


def randomized_svd(A, k, r = 2, q = 0, range_method = 'qr', proportion = None, oversamples = None):
    '''Compute the approximate randomized SVD of matrix A using either
       the QR method or the randomized range finder method.
       
        Parameters:
            A (ndarray) : mxn input matrix
            k (int) : target rank
            r (int) : number of subspace iterations
            q (int) : number of power iterations (default is 0)
            range_method (str) : method to compute the range ('qr' or 'subspace_iter')
                qr : use the basic numpy randomized range finder
                subspace_iter : use the randomized subspace iteration method (Algorithm 4.4.2)
            proportion (float) or oversamples (int): the proportion of columns to keep in your projection or the number of oversamples to add
                pick one of these two parameters to set the size of the random projection matrix
                if both are None, defaults to oversamples = k
                if both are provided, proportion takes precedence
        Returns:
            U (ndarray) : mxk matrix of left singular vectors
            S (ndarray) : vector of singular values
            Vt (ndarray) : kxn matrix of right singular vectors
    '''
    m,n = A.shape
    
    if proportion == None and oversamples == None:
        Ω = np.random.randn(n, 2*k) 
    elif proportion is not None:
        Ω = np.random.randn(n,int(proportion * n))
    else:
        Ω = np.random.randn(n, k + oversamples)
    Y = A @ Ω
    for _ in range(q):
        Y = A @ (A.T @ Y)
        Y, _ = np.linalg.qr(Y)         # I think we need to orthonormalize at each step of power iter for stability
    if range_method == 'qr':
        Q,_ = np.linalg.qr(Y)
    elif range_method == 'subspace_iter':
        Q = random_subspace_iter(A, Y, r)
    B = Q.T @ A
    U, S, Vt = np.linalg.svd(B, full_matrices = False)
    U = Q @ U
    return U[:,:k], S[:k], Vt[:k,:]