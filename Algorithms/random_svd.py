import numpy as np


def randomized_svd(A, k, q = 0, range_method = 'qr'):
    '''Compute the approximate randomized SVD of matrix A using either
       the QR method or the randomized range finder method.
       
        Parameters:
            A (ndarray) : mxn input matrix
            k (int) : target rank
            q (int) : number of power iterations (default is 0)
            range_method (str) : method to compute the range ('qr' or 'subspace_iter')
                qr : use the basic numpy randomized range finder
                subspace_iter : use the randomized subspace iteration method (Algorithm 4.4.2)
        Returns:
            U (ndarray) : mxk matrix of left singular vectors
            S (ndarray) : vector of singular values
            Vt (ndarray) : kxn matrix of right singular vectors
    '''
    m,n = A.shape
    Ω = np.random.randn(n,2*k) # is 2k the best choice?
    for _ in range(q):
        Y = A @ (A.T @ Y)
    Y = A @ Ω
    if range_method == 'qr':
        Q,_ = np.linalg.qr(Y)
    elif range_method == 'subspace_iter':
        Q = random_subspace_iter(A, 2*k, q)
    B = Q.T @ A
    U, S, Vt = np.linalg.svd(B, full_matrices = False)
    U = Q @ U
    return U[:,:k], S[:k], Vt[:k,:]