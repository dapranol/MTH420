# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Lucia>
<MTH420>
<4/29/22>
"""


# Problem 1
import numpy as np
from matplotlib import pyplot as plt

def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    #raise NotImplementedError("Problem 1 Incomplete")
    n = 4 
    matrix = np.random.normal(size=(n,n))

    row_means = matrix.mean(axis=1)
                                                                                
    return np.var(row_means)

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    #raise NotImplementedError("Problem 1 Incomplete")

    #return np.array(var_of_means)
    n = range(100, 1000, 100)
    var_of_means(n)
    print (prob1())
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    #raise NotImplementedError("Problem 2 Incomplete")

    x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
    s = np.sin(x)
    c = np.cos(x)
    t = np.arctan(x)

    plt.plot(x, s)
    plt.show()
    plt.plot(x, c)
    plt.show()
    plt.plot(x, t)
    plt.show()

# Problem 3
def prob3(x):
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    #raise NotImplementedError("Problem 3 Incomplete")

    x = np.arange(-2, 6, 0.1)

    x_1 = np.arange(-2, 1, 0.1)
    x_2 = np.arange(1.1, 6, 0.1)

    f_1 = 1/(x_1 - 1)
    f_2 = 1/(x_2 - 1)

    plt.plot(x_1, f_1, 'm:', linestyle='--', linewidth=4)
    plt.plot(x_2, f_2, 'm:', linestyle='--', linewidth=4)

    plt.xlim(-2, 6)
    plt.ylim(-6, 6)

    plt.show()

# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    #raise NotImplementedError("Problem 4 Incomplete")

    #define funtions                                                            
    x = np.arange(0, 2*np.pi, 0.1)                                              
                                                                                
    f_1 = np.sin(x)                                                             
    f_2 = np.sin(2*x)                                                           
    f_3 = np.sin(x)*2                                                           
    f_4 = np.sin(2*x)*2                                                         
                                                                                
    #plot funtions                                                              
    ax1 = plt.subplot(221)                                                      
    ax1.plot(x, f_1, 'g:', linestyle='-')                                       
    ax1.set_title('sin(x)')                                                     
                                                                                
    ax2 = plt.subplot(222)                                                      
    ax2.plot(x, f_2, 'r:', linestyle='--')                                      
    ax2.set_title('sin(2x)')                                                    
                                                                                
    ax3 = plt.subplot(223)                                                      
    ax3.plot(x, f_3, 'b:', linestyle='--')                                      
    ax3.set_title('2sin(x)')                                                    
                                                                                
    ax4 = plt.subplot(224)                                                      
    ax4.plot(x, f_4, 'm:', linestyle=':')                                       
    ax4.set_title('2sin(2x)')                                                   
                                                                                
    plt.axis([0, 2*np.pi, -2, 2])                                               
    plt.suptitle('trig funtions')                                               
    plt.show()  


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    #raise NotImplementedError("Problem 6 Incomplete")
    
    #define the function
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = (np.sin(X) * np.sin(Y)) / (X * Y)

    #define subplots

    plt.subplot(121)
    plt.pcolormesh(X, Y, Z, cmap="viridis")
    plt.colorbar()
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-2 * np.pi, 2 * np.pi)

    plt.subplot(122)
    plt.contour(X, Y, Z, 10, cmap="coolwarm")
    plt.colorbar()

    plt.show()

if __name__ == "__main__":
    #Problem 1
    n = 4
    var_of_means(n)
    #problem 2
    prob2()

    #problem 3
    x = np.arange(-2, 6, 0.1)
    prob3(x)

    #problem 4
    prob4()


    #problem 6
    prob6()
