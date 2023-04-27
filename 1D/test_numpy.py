import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
# from LossRules import
plt.ion()

def F_HO(x):
    """Dynamics for harmonic oscillator"""

    p = x[:,0]
    q = x[:,1]
    F = np.vstack((-q,p)).T
    return F

def true_solution(x,T):
    return np.hstack((x[0][0]*np.cos(T) - x[0][1]*np.sin(T),
                      x[0][0]*np.sin(T) + x[0][1]*np.cos(T)))

err = []
hvals = []
for n in [100,1000,10000]:
    x = np.array([[0,-1]])
    T = 1

    t = np.linspace(0, T, n, dtype=np.double)[:,None]
    h = t[1][0]-t[0][0]
    t = t[:-1] + h/2

    #Evaluate the model
    sol = true_solution(x,t)
    S = F_HO(sol)

    err += [np.linalg.norm(true_solution(x,T) - x - h*np.sum(S,axis=0))]
    hvals += [h]

a,b = np.polyfit(np.log(hvals),np.log(err),1)
print(a,b)
plt.loglog(hvals,err)
plt.xlim((hvals[0],hvals[-1]))
plt.xlabel('h')
plt.ylabel('error')
#sol = sol.numpy()
#plt.plot(S[:,0])
#plt.plot(S[:,1])
#plt.show()
