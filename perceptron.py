import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    for i in range(len(X)):
        pred = prediction(X[i],W,b)
        if y[i]-pred == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-pred == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return W,b


def getY(slope, intercept,x):
    return (slope * x + intercept)

def getPerceptronLine(w1,w2,b):
    slope = -(b / w2) / (b / w1)
    intercept = -b / w2
    xx = [-0.5,1.5]
    yy = [getY(slope,intercept,-0.5),getY(slope,intercept,1.5)]
    return xx,yy

if __name__ == "__main__":
    import pandas as pd
    columns=['x1','x2','y']
    df=pd.read_csv("data.csv", header=None, names=columns)
    X=np.array(df.values[:,:2])
    y=np.array(df.values[:,2:])
    result = trainPerceptronAlgorithm(X,y)
    line = getPerceptronLine(result[0][0],result[0][1],result[1])
    xes = X[:,:1].reshape(-1)
    yes = X[:,1:].reshape(-1)
    cnt = sum(i < 1 for i in y.reshape(-1))
    plt.plot(xes[:cnt],yes[:cnt], 'ro')
    plt.plot(xes[cnt:],yes[:cnt], 'go')
    plt.plot(line[0],line[1])
    plt.xlabel('# tests')
    plt.ylabel('grade')
    plt.title('Perceptron')
    plt.grid(True)
    plt.savefig("result.png")
    plt.show()
