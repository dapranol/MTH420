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
    A = np.random.normal(size=(n, n))
    A_t = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])

    Ad = np.random.normal(size=(n, n))

    k = index(A, tol=1e-5)

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

    T_1, Q_1, k_1 = la.schur(A, sort=lambda x: abs(x) > tol)
    T_2, Q_2, k_2 = la.schur(A, sort= lambda x: abs(x) <= tol)

    #create change of basis matrix 'U'
    U = np.hstack((Q_1[:, :k_1], Q_2[:, :len(Q_2) - k_1]))

    U_i = la.inv(U)
    V = np.dot(U_i, np.dot(A, U))

    Z = np.zeros_like(A, dtype=float)

    if k_1 != 0:
        M_i = la.inv(V[:k_1, :k_1])
        Z[:k_1, :k_1] = M_i
    return np.real(np.dot(U, np.dot(Z, U_i)))

def laplacian(A):
    """Compute the Laplacian matrix of the adjacency matrix A,
    as well as the second smallest eigenvalue.

    Parameters:
        A ((n,n) ndarray) adjacency matrix for an undirected weighted graph.

    Returns:
        L ((n,n) ndarray): the Laplacian matrix of A
    """
    D = A.sum(axis=1)    # The degree of each vertex (either axis).
    return np.diag(D) - A

def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ((n,n) ndarray) The matrix where the ijth entry is the effective
        resistance from node i to node j.
    """
    #raise NotImplementedError("Problem 3 Incomplete")

    n = len(A)
    L = laplacian(A)
    ER = np.zeros_like(A,dtype = float)
    L_tilda = np.copy(L)
    I = np.eye(n)

    for j in range(n):
        L_tilda = np.copy(L)
        L_tilda[j, :] = I[j, :]
        D = drazin_inverse(L_tilda)
        ER[:, j] = np.diag(D)

    return ER - I


if __name__=="__main__":
#Problem 1
    A_t = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    Ad_t = np.array([[1, -3, 9, 81], [0, 1, -3, -18], [0, 0, 1, 3], [0, 0, 0, 0]])
    k_A = 1

    is_drazin(A_t, Ad_t, k_A)

    B_t = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])
    Bd_t = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])                                                                     
    k_B = 3 

    is_drazin(B_t, Bd_t, k_B)

#Problem 2

    A_t = np.array([[1, 3, 0, 0], [0, 1, 3, 0], [0, 0, 1, 3], [0, 0, 0, 0]])
    B_t = np.array([[1, 1, 3], [5, 2, 6], [-2, -1, -3]])

    print (drazin_inverse(A_t, tol=1e-4))
    print (drazin_inverse(B_t, tol=1e-4))

#Problem 3 
    A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    laplacian(A)
    print (effective_resistance(A))
