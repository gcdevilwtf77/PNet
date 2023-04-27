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
torch.set_default_dtype(torch.float64)

def F_HO(x):
    """Dynamics for harmonic oscillator"""

    p = x[:,0]
    q = x[:,1]
    F = torch.vstack((-q,p)).T
    return F

def true_solution(x,T):
    return torch.hstack((x[0][0]*torch.cos(T) - x[0][1]*torch.sin(T),
                      x[0][0]*torch.sin(T) + x[0][1]*torch.cos(T)))

err = []
hvals = []
for n in [100,1000,100000]:
    x = torch.tensor([[0,-1]])
    T = 1

    t = torch.linspace(0, T, n)[:,None]
    h = t[1]-t[0]
    t = t[:-1] + h/2

    #Evaluate the model
    sol = true_solution(x,t)
    S = F_HO(sol)

    err += [torch.norm(true_solution(x,torch.tensor(T)) - x - h*torch.sum(S,dim=0)).item()]
    hvals += [h.item()]

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
