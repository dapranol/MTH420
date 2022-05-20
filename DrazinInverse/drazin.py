# drazin.py
"""Volume 1: The Drazin Inverse.
<Lucia Daprano>
<MTH420>
<04/13/22>
"""

import numpy as np
from scipy import linalg as la


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Compute the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


# Problem 1
def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    #raise NotImplementedError("Problem 1 Incomplete")

    n = 4
    #A = np.random.normal(size=(n, n))
    A = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    Ad = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])

    #Ad = np.random.normal(size=(n, n))

    k = 1

    if (A * Ad == Ad * A, np.linalg.matrix_power(A, k + 1) * Ad == np.linalg.matrix_power(A, k), Ad * A * Ad == Ad):
        print ("True")
    else:
        print ("False")

# Problem 2
def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
       ((n,n) ndarray) The Drazin inverse of A.
    """
    #raise NotImplementedError("Problem 2 Incomplete")
    n = len(A)

    A = np.random.normal(size=(n, n))
    A.shape = (n, n)

    T_1, Q_1, k_1 = la.schur(A, sort=lambda x: abs(x) > tol)
    T_2, Q_2, k_2 = la.schur(A, sort= lambda x: abs(x) <= tol)

    #create change of basis matrix 'U'
    U = np.vstack((Q_1[:, :k_1], Q_2[:, :n - k_1]))

    U_i = np.linalg.inv(U)
    V = U_i * A * U

    Z = np.zeros(n)

    if k_1 != 0:
        np.linalg.inv(V) == np.linalg.inv(M)
        np.linalg.inv(M) == Z
    return U * Z * U_i

# Problem 3
def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    raise NotImplementedError("Problem 3 Incomplete")

if __name__=="__main__":
#Problem 1
    A_t = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    Ad_t = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])
    k_A = 1

    B_t = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])
    Bd_t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])                                                                     
    k_B = 3 

    is_drazin(B_t, Bd_t, k_B)

#Problem 2

    A_t = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])

    drazin_inverse(A_t, tol=1e-4)
