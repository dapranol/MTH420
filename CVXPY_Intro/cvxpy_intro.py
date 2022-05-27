# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Lucia Daprano>
<MTH420>
<5/24/2022>
"""
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #raise NotImplementedError("Problem 1 Incomplete")

    x = cp.Variable(3, nonneg = True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    A = np.array([1, 2, 0])
    B = np.array([0, 1, -4])
    C = np.array([2, 10, 3])

    constraints = [A @ x <= 3, B @ x <= 1, C @ x >= 12]

    problem = cp.Problem(objective, constraints)
    print(problem.solve())
    print(x.value)

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #raise NotImplementedError("Problem 2 Incomplete")

    x = cp.Variable(len(A.T))
    objective = cp.Minimize(cp.norm(x, 1))
    
    constraints = [A @ x == b]

    problem = cp.Problem(objective, constraints)
    print(problem.solve())
    print(x.value)

# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    
    p = cp.Variable(6, nonneg = True)
    c = np.array ([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(p.T @ c)
    
    A = np.array([[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 1], [-1, 0, -1, 0, -1, 0], [0, -1, 0, -1, 0, -1]])
    a_n = np.array([7, 2, 4, -5, -8])

    constraints = [A @ p.T <= a_n.T] 

    problem = cp.Problem(objective, constraints)
    print(problem.solve())
    print(p.value)

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #raise NotImplementedError("Problem 4 Incomplete")
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
    r = np.array ([3, 0, 1])
    x = cp.Variable(3)
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    
    print(problem.solve())
    print(x.value)

# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    x = cp.Variable(len(A.T), nonneg = True)
    objective = (cp.Minimize(cp.norm(A @ x - b, 2)))

    constraints = [cp.sum(x) == 1]
    
    problem = cp.Problem(objective, constraints)

    print(problem.solve())
    print(x.value)
# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 
    raise NotImplementedError("Problem 6 Incomplete")

if __name__=="__main__":
    #Problem 1
    prob1()

    #Problem 2
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])

    l1Min(A, b)

    #Problem 3
    prob3()

    #Problem 4
    prob4()

    #Problem 5
    A = np.array([[1, 2, 1, 1], [0, 3, -2, -1]])
    b = np.array([7, 4])

    prob5(A, b)
