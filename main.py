#Matplotlib and Numpy
import numpy as np
import matplotlib.pyplot as plt

#Chainer specific imports
from chainer import FunctionSet, Variable, optimizers, serializers
import chainer.functions as F
import chainer.links as L

#Data set generation
import genData

(x, y)=genData.genLinearData()

#Linear link from variable to variable
linear_function = L.Linear(1,1)

#Set x and y as chainer variables
x_var = Variable(x.reshape(1000,-1))
y_var = Variable(y.reshape(1000,-1))

#setup optimizer Stochastic Gradient Decent
optimizer = optimizers.MomentumSGD(lr=0.001)
optimizer.setup(linear_function)

#Forward pass function, uses data as input
#Returns Linear Function as output
def linear_forward(data):
    return linear_function(data)

def linear_train(train_data, train_target, n_epochs=200):
    for _ in range(n_epochs):
        #Get result of forward pass
        output = linear_forward(train_data)

        #Calculate the loss between training data and target data.
        loss = F.mean_squared_error(train_target,output)

        #zero all gradients before using
        linear_function.zerograds()

        #Calculate and update gradients
        loss.backward()

        #Use optimizer to move params to values that reduce loss.
        optimizer.update()


plt.scatter(x,y, alpha=0.5)
for i in range(150):
    linear_train(x_var, y_var,n_epochs=5)
    y_pred = linear_forward(x_var).data
    plt.plot(x, y_pred, color = plt.cm.cool(i/150.), alpha = 0.4, lw = 3)

slope = linear_function.W.data[0,0]
intercept = linear_function.b.data[0]
plt.title("Final Line: {0:.3}x + {1:.3}".format(slope, intercept))
plt.xlabel('x')
plt.ylabel('y')
plt.show()
