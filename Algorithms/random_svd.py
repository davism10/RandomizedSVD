import numpy as np

def random_range_finder(A : NDArray, j : int):
    ''' Given an mxn matrix A and an integer j, this scheme computes an mxj
        orthonormal matrix Q whose range approximates the range of A.

        Parameters:
            A (ndarray) : mxn matrix
            j (int) : the desired number of columns in Q
        Returns:
            Q (ndarray) : mxj matrix with range(Q) approximately equal to range(A)
    '''
    m, n = A.shape        # Get the dimensions of A

    W = randn(n, j)       # Draw an nxj Gaussian random matrix W 
    Y = A @ W             # Form the mxj matrix Y = AW 
    Q, R = qr(Y)          # Construct the QR decomposition Y = QR

    return Q

def randomized_svd(A, k, q = 0, range_method = 'qr'):
    '''Compute the approximate randomized SVD of matrix A using either
       the QR method or the randomized range finder method.
       
        Parameters:
            A (ndarray) : mxn input matrix
            k (int) : target rank
            q (int) : number of power iterations (default is 0)
            range_method (str) : method to compute the range ('qr' or 'randomized') 
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
    if range_method == 'randomized':
        Q = randomized_range_finder(A, k)
    B = Q.T @ A
    U, S, Vt = np.linalg.svd(B, full_matrices = False)
    U = Q @ U
    return U[:,:k], S[:k], Vt[:k,:]