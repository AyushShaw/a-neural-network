import numpy as np

# Train data set
inpts = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
lable = np.array([[0], [1], [1], [1]])


def sigmoid(x):
    return (1/(1+np.exp(x)))


def sigdev(x):
    return x*(1-x)



