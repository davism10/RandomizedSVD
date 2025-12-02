import numpy as np

def randomized_svd(A, k, q = 1, range_method = 'qr', j = None):
    m,n = A.shape
    Ω = np.random.randn(n,2*k)
    Y = (A@A.T)**q @ A @ Ω
    Q,R = np.linalg.qr(Y)
    B = Q.T @ A
    if range_method == 'qr':
        Q,R = np.linalg.qr(Y)
    if range_method == 'randomized':
        Q = randomized_range_finder(A, j)
    U, S, Vt = np.linalg.svd(B, full_matrices = False)
    U = Q @ U
    return U[:,:k], S[:k], Vt[:k,:]