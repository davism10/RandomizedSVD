import numpy as np 
from numpy.linalg import qr 
from numpy.random import randn
from numpy.typing import NDArray

def random_range_finder(A : NDArray, j : int) -> NDArray:
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


def random_subspace_iter(A : NDArray, j : int, q : int) -> NDArray:
    ''' Given an mxn matrix A and integers j and q, this algorithm computes
        an mxj orthonormal matrix Q whose range approximates the range of A.

            Parameters:
                A (ndarray) : mxn matrix
                j (int) : specifies the desired number of columns in Q 
                q (int) : specifies desired number of iterations to perform

            Returns:
                Q (ndarray) : mxj matrix with range(Q) approximately equal to range(A)
    '''
    m, n = A.shape          # Get dimensions of A
    W = randn(n, j)         # Draw an nxj standard Gaussian matrix W

    # Form Y0 = AW and compute its QR factorization Y0 = Q0*R0
    Y_old = A @ W 
    Q_old, R_old = qr(Y_old)

    # iterate from 1 to q
    for _ in range(1, q+1):
        Y_approx = A.conj() @ Q_old
        Q_approx, R_approx = qr(Y_approx)
        Y_new = A @ Q_approx 
        Q_new, R_new = qr(Y_new)

        # update values
        Q_old = Q_new.copy()
    
    # set Q = Qq
    return Q_new

def random_svd_power_iter(A : NDArray, k : int, q : int = 1) -> NDArray:
    ''' Use Randomized SVD with power iteration to compute the SVD of the matrix A.
        This procedure computes an approximate rank-2k factorization USVh, 
        where U and V are orthonormal and S is nonnegative and diagonal. 

            Parameters:
                A (ndarray) : mxn matrix
                k (int) : target number of singular vectors
                q (int) : exponent, defaults to 1
            Returns:
                U (ndarray) : matrix of left singular vectors
                S (ndarray) : vector of singular values
                Vh (ndarray) : vector of right singular vectors
    '''
    pass 
    # STAGE A 
    # 1. Generate an n x 2k Gaussian test matrix W
    # 2. Form Y = ((AA*)^q)AW by multiplying alternately with A and A*
    # 3. Construct a matrix Q whose columns form an orthonormal basis for range(Y)

    # STAGE B
    # 4. Form B = Q*A
    # 5. Compute an SVD of the small matrix B = RSVh
    # 6. Set U = QR