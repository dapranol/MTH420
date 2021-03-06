# standard_library.py
"""Python Essentials: The Standard Library.
<Lucia>
<MTH420>
<5/15/22>
"""


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    return (min(L), max(L), sum(L) / len(L))   


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    int_1 = {1: 1, 2: 2}                                                        
    int_2 = int_1                                                               
    int_2[1] = 4                                                                
    print('numbers:', int_1 == int_2)                                                       
        #integral is mutable                                                    
    str_1 = "Python is cool"                                                    
    str_2 = str_1                                                               
    str_2 = "C"                                                                 
    print('strings:', str_1 == str_2)                                                       
        #string is mutable                                                      
    list_1 = ["rock", "paper", "scissors"]                                      
    list_2 = list_1                                                             
    list_2[2] = "fire"                                                          
    print('lists:', list_1 == list_2)                                                     
        #list is mutable                                                        
    tuple_1 = [1, 'a', 7, 'birthdays']                                          
    tuple_2 = tuple_1                                                           
    tuple_2 = (3, 4, 5)                                                         
    print('tuples:', tuple_1 == tuple_2)                                                   
        #tuple is mutable                                                       
    set_1 = [5, 6, 8]                                                           
    set_2 = set_1                                                               
    set_2 = [4, 5, 6]                                                           
    print('sets:', set_1 == set_2)                                                       
        #set is mutable

# Problem 3
import calculator as calc
import math
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than sum(), product() and sqrt that are 
    imported from your 'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """
    #raise NotImplementedError("Problem 3 Incomplete")
    hypot =  math.sqrt (calc.sum_numbers(calc.product_numbers(a, a),calc.product_numbers(b, b)))
    return hypot
# Problem 4
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
from itertools import combinations, permutations, chain

def power_set(A):
    P = []
    for r in range(len(A) + 1):
        for combo in combinations(A, r):
            P.append(set(combo))
    return P

    #raise NotImplementedError("Problem 4 Incomplete")


if __name__ == "__main__":
    #problem 1
    L = [1, 2, 3, 4, 5]
    print (prob1(L))

    #problem 2
    prob2()

    #problem 3
    a = 3
    b = 4
    print (hypot(a, b))

    #problem 4
    At = list({1, 2, 3})
    print (power_set(At))
