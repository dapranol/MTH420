# oneD_optimization.py
"""Volume 2: One-Dimensional Optimization.
<Lucia>
<MTH420>
<May 30, 2022>
"""
import math
import itertools
import numpy as np
# Problem 1
def golden_section(f, a, b, tol=1e-5, maxiter=15):
    """Use the golden section search to minimize the unimodal function f.

    Parameters:
        f (function): A unimodal, scalar-valued function on [a,b].
        a (float): Left bound of the domain.
        b (float): Right bound of the domain.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    
    x0 = (a + b)/ 2
    u = (1 + math.sqrt(5))/2
    
    for i in range(1, maxiter):
        c = (b - a)/ u
        a_tilda = b - c
        b_tilda = a + c
        if f(a_tilda) <= f(b_tilda):
            b = b_tilda
        else:
            a = a_tilda
        x1 = (a + b)/2
        if abs(x0 - x1) < tol:
            break
        x0 = x1

    return x1

# Problem 2
def newton1d(df, d2f, x0, tol=1e-5, maxiter=15):
    """Use Newton's method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        d2f (function): The second derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    x = x0

    for k in range(0, maxiter):
        x[k + 1] = x[k] - df/d2f
        if abs(x[k] -x[k-1]) < tol:
            break
    return x

# Problem 3
def secant1d(df, x0, x1, tol=1e-5, maxiter=15):
    """Use the secant method to minimize a function f:R->R.

    Parameters:
        df (function): The first derivative of f.
        x0 (float): An initial guess for the minimizer of f.
        x1 (float): Another guess for the minimizer of f.
        tol (float): The stopping tolerance.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): The approximate minimizer of f.
        (bool): Whether or not the algorithm converged.
        (int): The number of iterations computed.
    """
    raise NotImplementedError("Problem 1 Incomplete")


# Problem 4
def backtracking(f, Df, x, p, alpha=1, rho=.9, c=1e-4):
    """Implement the backtracking line search to find a step size that
    satisfies the Armijo condition.

    Parameters:
        f (function): A function f:R^n->R.
        Df (function): The first derivative (gradient) of f.
        x (float): The current approximation to the minimizer.
        p (float): The current search direction.
        alpha (float): A large initial step length.
        rho (float): Parameter in (0, 1).
        c (float): Parameter in (0, 1).

    Returns:
        alpha (float): Optimal step size.
    """
    raise NotImplementedError("Problem 1 Incomplete")



if __name__ == "__main__":

    #Problem 1

    f = lambda x : np.exp(x) - 4 * x
    a = 0
    b = 3
    print (golden_section(f, a, b, tol=1e-5, maxiter=15))

    #Problem 2

    df = lambda x : 2*x + 5*np.cos(5*x)
    d2f = lambda x : 2 - 25*np.sin(5*x)
    x0 = 0

    print (newton1d(df, d2f, x0, tol=1e-5, maxiter=15))
