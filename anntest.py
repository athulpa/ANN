
import numpy as np
import ANN
import keras
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

(x_train, y_train), (x_test, y_test) = mnist.load_data()

xtr = x_train.reshape((-1, 784))
xts = x_test
ytr = to_categorical(y_train, 10)
yts = to_categorical(y_test, 10)

def f(n):
    return xts[n], yts[n]

#m = ANN.ANN([784, 100, 10])

def Cost(m):
    ypr = m.fpbatch(x_test.reshape((-1, 784)))
    return np.mean(np.sum(np.square(ypr-yts), 1))/2
    
def ConfMat(m):
    
    CM = np.zeros((10,10), dtype=np.int64)
    ypr = m.fpbatch(x_test.reshape((-1, 784)))
    ypr = np.argmax(ypr, 1)
    for i in range(len(ypr)):
        CM[ypr[i], y_test[i]] += 1
    return CM
    
def runit(m, pieces=60, ep=20, lr=0.003):
    bsize = int(60000/pieces)
    for j in range(ep):
        ch = [ ]
        print('\nEpoch', j+1)
        print('Starting Cost:', Cost(m))
        for i in range(pieces):
            m.bpbatch(xtr[bsize*i:bsize*(i+1)], ytr[bsize*i:bsize*(i+1)], lr, 0)
            ch.append(Cost(m))
        plt.plot(ch)
        plt.gca().set_ylim(0)
        plt.show()
        print('Avg Cost:', sum(ch)/pieces)
        print('Ending Cost:', ch[-1])
        plt.clf()
        
def acc(ConfMat):
    t=np.sum(ConfMat)
    dsum=0
    for i in range(ConfMat.shape[0]):
        dsum += ConfMat[i, i]    
    return dsum/t

# Calculates n Geometric Means b/w a and b and returns them in a list.
def GMs(a, b, n):
    r = (b/a)**(1/(n+1))
#    print(r)
    return [a * r**(i+1) for i in range(n)]
    