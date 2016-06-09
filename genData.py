#Matplotlib and Numpy
import numpy as np
import matplotlib.pyplot as plt


# Generate linearly related datasets x and y.
# returns tuple x, y
def genLinearData():
    x = 30*np.random.rand(1000).astype(np.float32)
    y = 7*x+10
    y += 10*np.random.randn(1000).astype(np.float32)
    plt.scatter(x,y)
    plt.xlabel('x')
    plt.ylabel('y')
    return (x, y)
