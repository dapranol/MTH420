# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Lucia>
<MTH420>
<05/03/22>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
import scipy.linalg as la 
import math 
# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """

    Q, R = la.qr(A, mode="economic")

    x = la.solve_triangular(R, (Q.T @ b))
    print("b", b)
    print("x", x)
    print("Q", Q)
    print("R", R)

    return x

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
    #raise NotImplementedError("Problem 2 Incomplete"
    data = np.load("housing.npy")

    plt.scatter(data[:,0], data[:,1])                                           
    plt.plot(data[:,0], data[:,1], 'k,')                                        
                                                                                
    size = len(data[:,0])                                                       
    O = np.ones((size, 1))                                                      
    A = np.column_stack((list(data[:, 0]),O))
    t = data[:, 0]
    b = data[:, 1]                  
    
    c = least_squares(A, b)
    plt.plot(t, c[0]*t + c[1])
    plt.show()
    print (c)

# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    data = np.load("housing.npy")

    A_3 = np.vander(data[:, 0], 4)
    A_6 = np.vander(data[:, 0], 7)
    A_9 = np.vander(data[:, 0], 10)
    A_12 = np.vander(data[:, 0], 13)

    b = data[:, 1]

    w = la.lstsq(A_3, b)[0]
    x = la.lstsq(A_6, b)[0]
    y = la.lstsq(A_9, b)[0]
    z = la.lstsq(A_12, b)[0]

    l = np.linspace(0, 50)
    t = data[:, 0] 
    fig, axs = plt.subplots(2, 2)
    plt.subplot(2, 2, 1).plot(t, w[0]*t**3 + w[1]*t**2 + w[2]*t + w[3])
    plt.scatter(data[:, 0], data[:, 1])
    plt.subplot(2, 2, 2).plot(t, x[0]*t**6 + x[1]*t**5 + x[2]*t**4 + x[3]*t**3 + x[4]*t**2 + x[5]*t + x[6])
    plt.scatter(data[:, 0], data[:, 1])
    plt.subplot(2, 2, 3).plot(t, y[0]*t**9 + y[1]*t**8 + y[2]*t**7 + y[3]*t**6 + y[4]*t**5 + y[5]*t**4 + y[6]*t**3 + y[7]*t**2 + y[8]*t + y[9])
    plt.scatter(data[:, 0], data[:, 1])
    plt.subplot(2, 2, 4).plot(t, z[0]*t**12 + z[1]*t**11 + z[2]*t**10 + z[3]*t**9 + z[4]*t**8 + z[5]*t**7 + z[6]*t**6 + z[7]*t**5 + z[8]*t**4 + z[9]*t**3 + z[10]*t**2 + z[11]*t + z[12])
    plt.scatter(data[:, 0], data[:, 1])
    
    plt.show()

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    raise NotImplementedError("Problem 6 Incomplete")

if __name__=="__main__":
     #problem 1                                                                  
    n = 3 # int(input('enter n:'))                                              
    m = 4 # int(input('enter m (n <= m):'))                                     
                                                                                
    A = np.random.normal(size=(m, n))                                           
    b = np.random.normal(size=(m, 1))                                           
                                                                                
    #problem 2                                                              
    line_fit()

    #problem 3
    polynomial_fit()
