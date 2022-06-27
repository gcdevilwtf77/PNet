import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator(p0,q0,t):
    """Finite difference solver for harmonic oscillator"""

    n = len(t)

    p = np.zeros(n)
    q = np.zeros(n)
    p[0],q[0] = p0,q0

    for i in range(1,n):
        h = (t[i]-t[i-1])
        p[i] = p[i-1] - h*q[i-1]
        q[i] = q[i-1] + h*p[i-1]

    return p,q

class Net(nn.Module):
    def __init__(self, n_hidden=100):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden,n_hidden)
        self.fc4 = nn.Linear(n_hidden,2)

    def forward(self, state):
        x = state
        # x = torch.hstack((state,t))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x  #(p,q)

def F_HO(x):
    """Dynamics for harmonic oscillator"""

    p = x[:,0]
    q = x[:,1]
    Fs = torch.vstack((-q,p)).T
    return Fs

Model = torch.load('EulerCorrection.pt', map_location=torch.device('cpu'))

#Initial conditions
p0,q0 = 0,1

Model.eval()
plt.figure()
with torch.no_grad(): #Tell torch to stop keeping track of gradients
    t = torch.linspace(0, 10, 10000, dtype=torch.float)
    dt = torch.reshape(t[1]-t[0],(1,1))
    S = torch.tensor([[p0,q0]],dtype=torch.float)
    p = torch.zeros(len(t))
    q = torch.zeros(len(t))
    h = 0.001
    p[0],q[0] = p0,q0
    for i in range(1,len(t)):
        S = S + h*F_HO(S) + h**2/2*F_HO(Model(S))
        p[i] = S[0,0]
        q[i] = S[0,1]

    plt.plot(t, p, label="Neural Net: p")
    plt.plot(t, q, label="Neural Net: q")

t = torch.linspace(0, 10, 10000, dtype=torch.float)
p,q = harmonic_oscillator(p0,q0,t)
plt.plot(t, p, label="Finite Diff: p")
plt.plot(t, q, label="Finite Diff: q")
plt.legend()
plt.show()