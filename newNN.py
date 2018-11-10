
# Use this file with second round of custm ANN class.

import numpy as np
from ANN import ANN
from IDS import np_getalldat
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt


convDict = {
'Iris-setosa':0,
'Iris-virginica':1,
'Iris-versicolor':2
}

# Uses rept to do a basic_fit 100 times, and performs all this for size of hidden layer = 'start' : 'stop'
def linsearch(start=1, stop=15):
    accs = [ ]
    cmtrs = [ ]
    cmtss = [ ]
    for i in range(start, stop+1):
        print('\n\nHIDDEN LAYERS SHAPE: [..', i,'..]', sep='')
        vals = rept([i], 100)
        accs.append((vals[0],vals[1]))
        cmtrs.append(vals[2])
        cmtss.append(vals[3])
    return accs, cmtrs,cmtss


# Uses rept to do a basic_fit 100 times, and performs all this for 2 hidden layers,
# whose sizes vary according to the inputs to this fn.
def quadsearch(h1_start=1, h1_stop=10, h2_start=1, h2_stop=10):
    accs = np.zeros((h1_stop+1-h1_start, h2_stop+1-h2_start, 2))
    cmtrs = np.zeros((h1_stop+1-h1_start, h2_stop+1-h2_start, 3, 3))
    cmtss = np.zeros((h1_stop+1-h1_start, h2_stop+1-h2_start, 3, 3))
    for i in range(h1_start, h1_stop+1):
        for j in range(h2_start, h2_stop+1):
            print('\n\nHIDDEN LAYERS SHAPE: [..', i,', ', j, '..]', sep='')
            vals = rept([i, j], 100)
            accs[i-h1_start][j-h2_start] = np.array((vals[0],vals[1]))
            cmtrs[i-h1_start][j-h2_start] = vals[2]
            cmtss[i-h1_start][j-h2_start] = vals[3]
    return accs, cmtrs, cmtss

# Repeats the run 'cycles' no of time to get the average metrics.
def rept(ls, cycles=25, alpha = 0.003, MaxIter=1000):
# Initialization is done by calling for cnt=0 manually.    
    (cmtr, acctr), (cmts, accts) = basic_fit(ls, alpha=alpha, MaxIter=MaxIter) 
    
    for cnt in range(1, cycles):
        (cmtr_i, acctr_i), (cmts_i, accts_i) = basic_fit(ls, alpha=0.003, MaxIter=1000)    
        try:
            cmts += cmts_i
        except ValueError:
            print('ValueError')
            return cmts, cmts_i
        cmtr += cmtr_i
        acctr += acctr_i
        accts += accts_i
    cmtr = cmtr/cycles
    cmts = cmts/cycles
    acctr/=cycles
    accts/=cycles
    print('\nFor Training Data:')
    print('accuracy:', round(100*acctr, 2), '%')
    print('Confusion Matrix:\n', cmtr)
    print('\nFor Test Data:')
    print('accuracy:', round(100*accts, 2), '%')
    print('Confusion Matrix:\n', cmts)
    return acctr, accts, cmtr, cmts
    

def basic_fit(ls, alpha=0.003, MaxIter=1000):
    tr, ts = np_getalldat()
    
    nn = ANN([4] + ls + [3])
    
    tr, ts = np_getalldat()
    
    trins = [ ]
    trouts = [ ]
    for i in range(len(tr)):
        trins.append(tr[i][0])
        trouts.append(convDict[tr[i][1]])
    trouts = to_categorical(trouts, 3)
    trins = np.array(trins)
    tsins = [ ]
    tsouts = [ ]
    for i in range(len(ts)):
        tsins.append(ts[i][0])
        tsouts.append(convDict[ts[i][1]])
    tsouts = to_categorical(tsouts, 3)
#    print(tsouts)
    tsins = np.array(tsins)    
#    print(tsins)
    
    for i in range(MaxIter):
        nn.bpbatch(trins, trouts, alpha, 0)
        
    return confmat(nn, trins, trouts, False), confmat(nn, tsins, tsouts, False)

# outs should be in categorical form.
def confmat(nn, ins, outs, verbose=True):
    preds = nn.fpbatch(ins)
#    print(preds)
    preds = np.argmax(preds, axis=1)
    outs = np.argmax(outs, axis=1)
#    print(preds, '\n', outs, '\n')
#    input()
    
    cm = np.zeros((max(outs)+1, max(outs)+1), dtype=np.int64)
    for i in range(len(preds)):
        cm[preds[i]][outs[i]]+=1
    TPos = 0
    for i in range(max(preds)+1):
        TPos+=cm[i][i]
    acc = TPos/np.sum(cm)
    
    if(verbose is True):
        print('accuracy:', round(100*acc, 2), '%')
        print('Confusion Matrix:\n', cm)
    
    return cm, acc
        