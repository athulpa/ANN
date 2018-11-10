
import numpy as np

# When giving theta inputs to neural network, always make sure that every theta is 2-dim, 
# even if its output is only 1 number. For example, [[1, 2, 3]] is correct, [1, 2, 3] is wrong.

class ANN:
    def __init__(self, sl):
        self.num_layers = len(sl)
        self.num_inputs = sl[1]
        self.num_outputs = sl[-1]
        
        self.theta = [ ]
        self.activation = ANN.sigmoid
        for i in range(self.num_layers-1):
            self.theta.append(np.random.random((sl[i+1], sl[i]+1))-0.5)
#            self.theta = [np.array([[-25,   5,   5,  20,  20],
#                                    [-25,  20,  20,   5,   5]], dtype=np.float64), np.array([[-10,  20,  20]], dtype=np.float64)]
        return
    
    def sigmoid(arr):
        return 1/(1+np.exp(-arr))
    
    def relu(arr):
        return np.where(arr<0, np.zeros(arr.shape), arr)
        
    def fpgt(self, arr):
#        print('inside')
        for th in self.theta:   
            arr = np.insert(arr, 0, 1)
#            print(th, arr)
            arr = np.matmul(th, arr)
#            print(arr)
            arr = self.activation(arr)
#            print(arr.shape, '\n')
#        print('leaving..')
        return arr
        
# Returns the D values computed on self.theta with the given single exmaple of ins and outs.    
    def bpgt(self, ins, outs):
        a = [ ins ]
        for th in self.theta:
            a[-1] = np.insert(a[-1], 0, 1)
            dotted = np.matmul(th, a[-1])
            a.append(self.activation(dotted))

        d = [ a[-1]-outs ]
        for i in np.flip(np.arange(1, self.num_layers-1), axis=0):   
            d.insert(0, (np.matmul(self.theta[i].T, d[0])*a[i]*(1-a[i]))[1:])
            
        D = [ ]
        for i in range(self.num_layers-1):
            D.append(d[i].reshape(-1, 1)*a[i])
        return D
    
    # arrs is expected to be 2D only, with elements arranged along axis-0.
    def fpbatch(self, arrs):
        for th in self.theta:
            arrs = np.insert(arrs, 0, 1, 1)
            arrs = np.matmul(arrs, th.T)
            arrs = self.activation(arrs)
        return arrs

#    Performs one step of the backprop update (i.e. 1 step of GD), given m examples of i/o values.
#    Examples must be placed down the column, ie along axis 0.        
    def bpbatch(self, ins, outs, alpha=0.05, lmd=0.1):
        a = [ins]
        for th in self.theta:
#            print(a[-1].shape)
            a[-1] = np.insert(a[-1], 0, 1, 1)
#            print(a[-1].shape)
#            print(th.T.shape)
            dotted = np.matmul(a[-1], th.T)
#            print(dotted.shape)
            a.append(self.activation(dotted))
#            print(a[-1].shape, '\n')
#            input()
        
        d = [ a[-1]-outs ]
        for i in np.flip(np.arange(1, self.num_layers-1), axis=0):
            d.insert(0, (np.matmul(d[0], self.theta[i])*a[i]*(1-a[i]))[:, 1:])
            
#        for dee in d:
#            print(dee.shape)
#        print('\n')
            
#        for ayy in a:
#            print(ayy.shape)
#        print('\n')            
            
        for i in range(self.num_layers-1):
#            print(i)
            Dtmp = np.expand_dims(d[i], 2)*np.expand_dims(a[i], 1)
            Dtmp = np.sum(Dtmp, axis=0)
#            print(Dtmp.shape, self.theta[i].shape, '\n')
            Dtmp += lmd*self.theta[i]
            self.theta[i] -= alpha*Dtmp
        return

