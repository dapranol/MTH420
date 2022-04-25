# python_intro.py
"""Python Essentials: Introduction to Python.
<Lucia>
<MTH420>
<4/8/22>
"""

#problem 1
pi = 3.14159                                                                    
x = 4/3                                                                         
r = 10                                                                          
V = x * pi * r**3                                                               
print (V)                                                                       
                                                                                
import numpy as np                                                              
#problem 3                                                                      
def sphere_volume(r):                                                           
    pi = 3.14159                                                                
                                                                                
    return 4/3* pi * r**3                                                       
                                                                                
#problem 4                                                                      
def prob4():                                                                    
    A = np.array([ [3, -1, 4],[1, 5, -9]])                                      
    B = np.array([ [2, 6, -5, 3],[5, -8, 9, 7],[9, -3, -2, -3]])                
                                                                                
    return A @ B # np.dot(A,B)                                                  
                                                                                
#problem 5                                                                      
def tax_liability(income):                                                      
    Limit1 = 9875                                                               
    Limit2 = 40125                                                              
    Limit3 = 85525                                                              
                                                                                
    Rate1 = .1                                                                  
    Rate2 = .12                                                                 
    Rate3 = .22                                                                 
                                                                                
    if income < Limit1:                                                         
        tax = (Rate1 * income)                                                  
    elif income < Limit2:                                                       
        tax = (Rate1 * Limit1 +Rate2 * (income - Limit1))                       
    elif income < Limit3:                                                       
        tax = Rate1 * Limit1 +Rate2 * (Limit2 - Limit1) + Rate3 * (income - Limit2)
    else:                                                                       
        tax = Rate1 * Limit1 + Rate2 * (Limit2 - Limit1) + Rate3 * (Limit3 - Limit2) * (income - Limit3)
    return tax

#problem 6

def prob6b():                                                                   
    A = np.array([1, 2, 3, 4, 5, 6, 7])                                         
    B = np.array([5, 5, 5, 5, 5, 5, 5])                                         
                                                                                
    return A @ B , A + B, 5 * A                                                 
                                                                                
if __name__ == "__main__":                                                      
    # problem 2
    print("Hello World!")                                                       
                                                                                
    #problem 3                                                                  
    radius = 10                                                                 
    V = sphere_volume(radius)                                                   
    print(V)                                                                    
                                                                                
    #problem 4                                                                  
    AB = prob4()                                                                
    print(AB)                                                                   
                                                                                
    #problem 5
    income = float(input("enter income"))
    T = tax_liability(income)                                                   
    print(T)                                                                    
                                                                                
    #problem 6                                                                                                                                
    print(prob6b())             


