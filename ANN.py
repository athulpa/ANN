
import numpy as np

class ANN:
    def __init__(self, sl):
        self.num_layers = len(sl)
        self.num_inputs = sl[1]
        self.num_outputs = sl[-1]
        
        self.theta = [ ]
        for i in range(self.num_layers-1):
            self.theta.append(np.random.random((sl[i+1], sl[i]+1)))
        return
    
    def sigmoid(arr):
        return 1/(1+np.exp(-arr))
    
    def relu(arr):
        return np.where(arr<0, np.zeros(arr.shape), arr)
        
    def fpgt(self, arr):
        for th in self.theta:           
            arr = np.matmul(th, arr)
        return arr
        
    def bpgt(self, ins, outs, alpha=0.05):
        z = [ ]
        a = [ ins ]
        for th in self.theta:
            a[-1] = np.insert(a[-1], 0, 1)
            z.append(np.matmul(th, a[-1]))
            a.append(sigmoid(z[-1]))
        
        d = [ a[-1]-outs ]
        for i in np.flip(np.arange(1, self.num_layers-1), axis=0):   
            d.insert(0, np.matmul(th[i].T, d[0])*a[i]*(1-a[i]))
        D = [ ]
        for i in range(self.num_layers-1):
            D.append(d[i].reshape(-1, 1)*a[i])
        return D
    
    def fpbatch(self, arrs):
        pass
    
    def bpbatch(self, ins, outs, alpha=0.05, lmd=0):
        pass
            