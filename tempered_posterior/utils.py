import numpy as np

def pshape(arr):
    print(arr.shape)

def mse(a,b):
    assert(len(a.shape)==1) # ensure 1 dim vector
    assert(len(b.shape)==1) # ensure 1 dim vector
    
    return np.mean((a-b)**2)