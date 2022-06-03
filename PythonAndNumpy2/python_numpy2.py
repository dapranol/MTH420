# python_intro.py
"""Python Essentials: Introduction to Python.
<Lucia>
<MTH420>
<4/21/22>
"""

#Problem 1
def isolate(a, b, c, d, e):
    print(a, b, c, sep ='     ', end= '  ')
    print(d, e)

    #raise NotImplementedError("Problem 1 Incomplete")

#Problem 2
def first_half(string):
    #raise NotImplementedError("Problem 2 Incomplete")

    return(string[:len(string)//2 if len(string)%2 == 0                          
        else (((len(string)//2))+1):])

def backward(string):
    return(string[::-1])
    #raise NotImplementedError("Problem 2 Incomplete")

#Problem 3
def list_ops(my_list):
    my_list = ['bear', 'ant', 'cat', 'dog']
    my_list.append('eagle')
    my_list[1] = 'fox'
    my_list.remove('bear')
    my_list.sort(reverse=True)
    my_list.insert(my_list.index('eagle'),'hawk')
    my_list.append('hunter')
    return my_list

    #raise NotImplementedError("Problem 3 Incomplete")

#Problem 4
def alt_harmonic(n):
    
    """Return the partial sum of the first n terms of the alternating
    harmonic series. Use this function to approximate ln(2).
    """
    #raise NotImplementedError("Problem 4 Incomplete")

    return(sum([((-1)**(n+1))/n for n in range(1,500000)])) 
#Problem 5
def prob5(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    #raise NotImplementedError("Problem 5 Incomplete")
    mask = A < 0
    A[mask] = 0
    return(np.copy(A))
import numpy as np

#Problem 6
def prob6():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    A = np.array([[0, 2, 4], [1, 3, 5]])
    B = np.array([[3, 0, 0], [3, 3, 0], [3, 3, 3]])
    C = np.diag([ -2, -2, -2 ])
    I = np.eye(3)
    Z_1 = np.zeros((3, 3))
    Z_2 = np.zeros((2, 2))
    Z_3 = np.zeros((2, 3))
    Matrix_1 = np.vstack((Z_1, A, B))
    Matrix_2 = np.vstack((A.T, Z_2, Z_3.T))
    Matrix_3 = np.vstack((I, Z_3, C))
    return np.hstack((Matrix_1, Matrix_2, Matrix_3))

def prob7(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    raise NotImplementedError("Problem 7 Incomplete")


def prob8():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    #raise NotImplementedError("Problem 8 Incomplete")
    def largest_adj(m):
        def mul_max(largest, *entries):
            prod = np.prod(entries)

            if prod > largest:
                largest = prod

            return largest

        largest = 0
        for r in range(len(m)):
            for c in range(len(m[0])):
                if r < len(m) - 3 and c < len(m[0]) - 3:
                    # Get diagonal top left
                    largest = mul_max(largest, *[m[r+i][c+i] for i in range(4)])

                    # Get diagonal top right
                    largest = mul_max(largest, m[r+3][c], m[r+1][c+2], m[r+2][c+1], m[r][c+3])

                if r < len(m) - 3:
                    # Get vertical products
                    largest = mul_max(largest, *[m[r+i][c] for i in range(4)])

                if c < len(m) - 3:
                    # Get horizontal products
                    largest = mul_max(largest, *[m[r][c+i] for i in range(4)])

        return largest

    # import the matrix called m
    m = np.load("grid.npy")
    print(largest_adj(m))
if __name__ == "__main__":

#problem 1
    isolate(1, 2, 3, 4, 5)

#problem 2
    string = 'string'
    print(first_half(string))                          
    print(backward(string))
#problem 3
    my_list = ['bear', 'ant', 'cat', 'dog']
    print (list_ops(my_list)) 
    
#problem 4
    print(sum([((-1)**(n+1))/n for n in range(1,500000)]))
#problem 5
    A = np.array([[3, -1, 4], [-1, 5, -9]])
    print(prob5(A))

#problem 6
    print(prob6())
#problem 8
    prob8()
